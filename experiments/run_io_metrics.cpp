// Starling I/O Metrics Experiment
// Computes: Waste Ratio, Overlap Ratio, Read Amplification, Block Access
// details Outputs 3 CSV files in experiments/results/

#include <atomic>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <omp.h>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <boost/program_options.hpp>

#include "pq_flash_index.h"
#include "aux_utils.h"
#include "index.h"
#include "math_utils.h"
#include "memory_mapper.h"
#include "partition_and_pq.h"
#include "timer.h"
#include "utils.h"
#include "percentile_stats.h"

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include "linux_aligned_file_reader.h"
#endif

namespace po = boost::program_options;

// Per-query metrics computed from instrumentation data
struct QueryMetrics {
  unsigned query_id;
  unsigned nodes_visited;
  unsigned nodes_loaded;
  unsigned nodes_wasted;
  double   waste_ratio;
  double   read_amplification;
  double   mean_overlap_ratio;
  unsigned n_ios;
  unsigned n_blocks_accessed;
  float    latency_us;
};

// Per-block detail for a query
struct BlockDetail {
  unsigned query_id;
  unsigned block_id;
  unsigned total_nodes_in_block;
  unsigned visited_nodes_in_block;
  unsigned wasted_nodes_in_block;
};

template<typename T>
int run_experiment(diskann::Metric&   metric,
                   const std::string& index_path_prefix,
                   const std::string& mem_index_path,
                   const std::string& result_dir, const std::string& query_file,
                   const std::string& disk_file_path,
                   const unsigned num_threads, const unsigned recall_at,
                   const unsigned beamwidth, const _u32 search_io_limit,
                   const _u64 L, const _u32 mem_L, const bool use_page_search,
                   const float use_ratio, const unsigned num_queries_to_run) {
  std::cout << "=== Starling I/O Metrics Experiment ===" << std::endl;
  std::cout << "Parameters:" << std::endl;
  std::cout << "  L (search list): " << L << std::endl;
  std::cout << "  Beamwidth: " << beamwidth << std::endl;
  std::cout << "  use_page_search: " << use_page_search << std::endl;
  std::cout << "  num_queries: " << num_queries_to_run << std::endl;

  // Load query vectors
  T*     query = nullptr;
  size_t query_num, query_dim, query_aligned_dim;
  diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim,
                               query_aligned_dim);
  std::cout << "Loaded " << query_num << " queries of dim " << query_dim
            << std::endl;

  unsigned actual_num_queries =
      std::min((unsigned) query_num, num_queries_to_run);
  std::cout << "Will run " << actual_num_queries << " queries" << std::endl;

  // Create index
  std::shared_ptr<AlignedFileReader> reader = nullptr;
  reader.reset(new LinuxAlignedFileReader());

  std::unique_ptr<diskann::PQFlashIndex<T>> _pFlashIndex(
      new diskann::PQFlashIndex<T>(reader, use_page_search, metric));

  int res = _pFlashIndex->load(num_threads, index_path_prefix.c_str(),
                               disk_file_path);
  if (res != 0) {
    std::cerr << "Failed to load index" << std::endl;
    return res;
  }

  // Load in-memory navigation graph if needed
  if (mem_L) {
    _pFlashIndex->load_mem_index(metric, query_dim, mem_index_path, num_threads,
                                 mem_L);
  }

  std::cout << "Index loaded successfully." << std::endl;
  std::cout << "  nnodes_per_sector: " << _pFlashIndex->get_nnodes_per_sector()
            << std::endl;
  std::cout << "  max_node_len: " << _pFlashIndex->get_max_node_len()
            << std::endl;
  std::cout << "  num_points: " << _pFlashIndex->get_num_points() << std::endl;
  std::cout << "  is_page_search: " << _pFlashIndex->is_page_search()
            << std::endl;

  _u64 nnodes_per_sector = _pFlashIndex->get_nnodes_per_sector();

  // Storage for results
  std::vector<QueryMetrics>             all_query_metrics(actual_num_queries);
  std::vector<std::vector<BlockDetail>> all_block_details(actual_num_queries);

  std::cout << "\nRunning queries..." << std::endl;

  for (unsigned qi = 0; qi < actual_num_queries; qi++) {
    if (qi % 100 == 0) {
      std::cout << "  Processing query " << qi << "/" << actual_num_queries
                << std::endl;
    }

    std::vector<uint64_t>         result_ids(recall_at);
    std::vector<float>            result_dists(recall_at);
    diskann::QueryStats           stats = {};
    diskann::QueryInstrumentation instr;

    if (use_page_search) {
      _pFlashIndex->page_search(query + (qi * query_aligned_dim), recall_at,
                                mem_L, L, result_ids.data(),
                                result_dists.data(), beamwidth, search_io_limit,
                                false, use_ratio, &stats, &instr);
    } else {
      _pFlashIndex->cached_beam_search(
          query + (qi * query_aligned_dim), recall_at, L, result_ids.data(),
          result_dists.data(), beamwidth, search_io_limit, false, &stats, mem_L,
          &instr);
    }

    // Deduplicate visited and loaded node lists
    std::unordered_set<unsigned> visited_set(instr.visited_node_ids.begin(),
                                             instr.visited_node_ids.end());
    std::unordered_set<unsigned> loaded_set(instr.loaded_node_ids.begin(),
                                            instr.loaded_node_ids.end());

    unsigned nodes_visited = visited_set.size();
    unsigned nodes_loaded = loaded_set.size();
    unsigned nodes_wasted = 0;
    for (unsigned nid : loaded_set) {
      if (visited_set.find(nid) == visited_set.end()) {
        nodes_wasted++;
      }
    }

    double waste_ratio = (nodes_loaded > 0)
                             ? (double) nodes_wasted / (double) nodes_loaded
                             : 0.0;
    double read_amp = (nodes_visited > 0)
                          ? (double) nodes_loaded / (double) nodes_visited
                          : 0.0;

    // Compute Overlap Ratio (OR) per visited node
    // Build neighbor map from instrumentation data
    std::unordered_map<unsigned, std::unordered_set<unsigned>> neighbor_map;
    for (auto& nn_pair : instr.node_neighbors) {
      neighbor_map[nn_pair.first].insert(nn_pair.second.begin(),
                                         nn_pair.second.end());
    }

    // Build block membership: for each block, which nodes are in it
    // Deduplicate accessed blocks
    std::unordered_set<unsigned> accessed_blocks_set(
        instr.accessed_block_ids.begin(), instr.accessed_block_ids.end());

    std::unordered_map<unsigned, std::vector<unsigned>> block_to_nodes;
    if (use_page_search) {
      const auto& gp_layout = _pFlashIndex->get_gp_layout();
      for (unsigned blk : accessed_blocks_set) {
        if (blk < gp_layout.size()) {
          block_to_nodes[blk] = gp_layout[blk];
        }
      }
    } else {
      for (unsigned blk : accessed_blocks_set) {
        std::vector<unsigned> nodes_in_blk;
        for (_u64 n = 0; n < nnodes_per_sector; n++) {
          unsigned nid = blk * nnodes_per_sector + n;
          if (nid < _pFlashIndex->get_num_points()) {
            nodes_in_blk.push_back(nid);
          }
        }
        block_to_nodes[blk] = nodes_in_blk;
      }
    }

    // For each visited node, compute overlap ratio with co-located nodes
    double   total_overlap = 0.0;
    unsigned overlap_count = 0;
    // Map each node to its block
    std::unordered_map<unsigned, unsigned> node_to_block;
    if (use_page_search) {
      const auto& id2page = _pFlashIndex->get_id2page();
      for (unsigned nid : visited_set) {
        if (nid < id2page.size()) {
          node_to_block[nid] = id2page[nid];
        }
      }
    } else {
      for (unsigned nid : visited_set) {
        node_to_block[nid] = nid / nnodes_per_sector;
      }
    }

    for (unsigned u : visited_set) {
      auto blk_it = node_to_block.find(u);
      if (blk_it == node_to_block.end())
        continue;
      unsigned blk = blk_it->second;

      auto btn_it = block_to_nodes.find(blk);
      if (btn_it == block_to_nodes.end())
        continue;
      const auto& colocated = btn_it->second;

      if (colocated.size() <= 1)
        continue;

      auto nbr_it = neighbor_map.find(u);
      if (nbr_it == neighbor_map.end())
        continue;
      const auto& nbrs = nbr_it->second;

      unsigned intersection = 0;
      for (unsigned co : colocated) {
        if (co == u)
          continue;
        if (nbrs.find(co) != nbrs.end()) {
          intersection++;
        }
      }
      double or_u = (double) intersection / (double) (colocated.size() - 1);
      total_overlap += or_u;
      overlap_count++;
    }

    double mean_overlap =
        (overlap_count > 0) ? total_overlap / overlap_count : 0.0;

    // Compute per-block details
    std::vector<BlockDetail> block_details;
    for (unsigned blk : accessed_blocks_set) {
      auto btn_it = block_to_nodes.find(blk);
      if (btn_it == block_to_nodes.end())
        continue;
      const auto& nodes_in_blk = btn_it->second;

      unsigned total_in_block = nodes_in_blk.size();
      unsigned visited_in_block = 0;
      for (unsigned nid : nodes_in_blk) {
        if (visited_set.find(nid) != visited_set.end()) {
          visited_in_block++;
        }
      }
      unsigned wasted_in_block = total_in_block - visited_in_block;

      BlockDetail bd;
      bd.query_id = qi;
      bd.block_id = blk;
      bd.total_nodes_in_block = total_in_block;
      bd.visited_nodes_in_block = visited_in_block;
      bd.wasted_nodes_in_block = wasted_in_block;
      block_details.push_back(bd);
    }

    // Store metrics
    QueryMetrics qm;
    qm.query_id = qi;
    qm.nodes_visited = nodes_visited;
    qm.nodes_loaded = nodes_loaded;
    qm.nodes_wasted = nodes_wasted;
    qm.waste_ratio = waste_ratio;
    qm.read_amplification = read_amp;
    qm.mean_overlap_ratio = mean_overlap;
    qm.n_ios = stats.n_ios;
    qm.n_blocks_accessed = accessed_blocks_set.size();
    qm.latency_us = stats.total_us;
    all_query_metrics[qi] = qm;
    all_block_details[qi] = block_details;
  }

  std::cout << "\nAll queries processed. Writing results..." << std::endl;

  // ===== OUTPUT 1: per_query.csv =====
  {
    std::string   path = result_dir + "/per_query.csv";
    std::ofstream ofs(path);
    ofs << "query_id,nodes_visited,nodes_loaded,nodes_wasted,waste_ratio,"
        << "read_amplification,mean_overlap_ratio,n_ios,n_blocks_accessed,"
           "latency_us"
        << std::endl;
    ofs << std::fixed << std::setprecision(6);
    for (auto& qm : all_query_metrics) {
      ofs << qm.query_id << "," << qm.nodes_visited << "," << qm.nodes_loaded
          << "," << qm.nodes_wasted << "," << qm.waste_ratio << ","
          << qm.read_amplification << "," << qm.mean_overlap_ratio << ","
          << qm.n_ios << "," << qm.n_blocks_accessed << "," << qm.latency_us
          << std::endl;
    }
    ofs.close();
    std::cout << "  Written: " << path << std::endl;
  }

  // ===== OUTPUT 2: block_access.csv =====
  {
    std::string   path = result_dir + "/block_access.csv";
    std::ofstream ofs(path);
    ofs << "query_id,block_id,total_nodes_in_block,visited_nodes_in_block,"
           "wasted_nodes_in_block"
        << std::endl;
    for (auto& bd_vec : all_block_details) {
      for (auto& bd : bd_vec) {
        ofs << bd.query_id << "," << bd.block_id << ","
            << bd.total_nodes_in_block << "," << bd.visited_nodes_in_block
            << "," << bd.wasted_nodes_in_block << std::endl;
      }
    }
    ofs.close();
    std::cout << "  Written: " << path << std::endl;
  }

  // ===== OUTPUT 3: summary.csv =====
  {
    double total_waste_ratio = 0, total_read_amp = 0, total_overlap = 0;
    double total_ios = 0;
    double total_visited = 0, total_loaded = 0, total_wasted = 0;
    double total_blocks = 0;
    double total_latency = 0;

    for (auto& qm : all_query_metrics) {
      total_waste_ratio += qm.waste_ratio;
      total_read_amp += qm.read_amplification;
      total_overlap += qm.mean_overlap_ratio;
      total_ios += qm.n_ios;
      total_visited += qm.nodes_visited;
      total_loaded += qm.nodes_loaded;
      total_wasted += qm.nodes_wasted;
      total_blocks += qm.n_blocks_accessed;
      total_latency += qm.latency_us;
    }

    double n = actual_num_queries;

    std::string   path = result_dir + "/summary.csv";
    std::ofstream ofs(path);
    ofs << std::fixed << std::setprecision(6);
    ofs << "metric,value" << std::endl;
    ofs << "num_queries," << actual_num_queries << std::endl;
    ofs << "L," << L << std::endl;
    ofs << "beamwidth," << beamwidth << std::endl;
    ofs << "use_page_search," << use_page_search << std::endl;
    ofs << "nnodes_per_sector," << nnodes_per_sector << std::endl;
    ofs << "mean_waste_ratio," << total_waste_ratio / n << std::endl;
    ofs << "mean_read_amplification," << total_read_amp / n << std::endl;
    ofs << "mean_overlap_ratio," << total_overlap / n << std::endl;
    ofs << "mean_ios_per_query," << total_ios / n << std::endl;
    ofs << "mean_nodes_visited," << total_visited / n << std::endl;
    ofs << "mean_nodes_loaded," << total_loaded / n << std::endl;
    ofs << "mean_nodes_wasted," << total_wasted / n << std::endl;
    ofs << "mean_blocks_accessed," << total_blocks / n << std::endl;
    ofs << "mean_latency_us," << total_latency / n << std::endl;
    ofs << "total_nodes_visited," << (unsigned) total_visited << std::endl;
    ofs << "total_nodes_loaded," << (unsigned) total_loaded << std::endl;
    ofs << "overall_waste_ratio,"
        << (total_loaded > 0 ? total_wasted / total_loaded : 0) << std::endl;
    ofs << "overall_read_amplification,"
        << (total_visited > 0 ? total_loaded / total_visited : 0) << std::endl;
    ofs.close();
    std::cout << "  Written: " << path << std::endl;
  }

  // Print summary to console
  std::cout << "\n=== SUMMARY ===" << std::endl;
  std::cout << std::fixed << std::setprecision(4);
  {
    double tw = 0, tr = 0, to = 0, ti = 0;
    for (auto& qm : all_query_metrics) {
      tw += qm.waste_ratio;
      tr += qm.read_amplification;
      to += qm.mean_overlap_ratio;
      ti += qm.n_ios;
    }
    double n = actual_num_queries;
    std::cout << "  Mean Waste Ratio:         " << tw / n << std::endl;
    std::cout << "  Mean Read Amplification:  " << tr / n << "x" << std::endl;
    std::cout << "  Mean Overlap Ratio:       " << to / n << std::endl;
    std::cout << "  Mean I/Os per query:      " << ti / n << std::endl;
  }

  diskann::aligned_free(query);
  return 0;
}

int main(int argc, char** argv) {
  std::string data_type, dist_fn, index_path_prefix, result_dir, query_file,
      disk_file_path, mem_index_path;
  unsigned num_threads, K, W, search_io_limit;
  unsigned mem_L, L, num_queries;
  bool     use_page_search;
  float    use_ratio;

  po::options_description desc{"Arguments"};
  try {
    desc.add_options()("help,h", "Print information on arguments");
    desc.add_options()("data_type",
                       po::value<std::string>(&data_type)->required(),
                       "data type <int8/uint8/float>");
    desc.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                       "distance function <l2/mips/cosine>");
    desc.add_options()("index_path_prefix",
                       po::value<std::string>(&index_path_prefix)->required(),
                       "Path prefix to the index");
    desc.add_options()("result_dir",
                       po::value<std::string>(&result_dir)
                           ->default_value("experiments/results"),
                       "Directory to save result CSV files");
    desc.add_options()("query_file",
                       po::value<std::string>(&query_file)->required(),
                       "Query file in binary format (.fbin)");
    desc.add_options()("recall_at,K",
                       po::value<uint32_t>(&K)->default_value(10),
                       "Number of neighbors to be returned");
    desc.add_options()("search_list,L",
                       po::value<unsigned>(&L)->default_value(100),
                       "Search list size");
    desc.add_options()("beamwidth,W", po::value<uint32_t>(&W)->default_value(4),
                       "Beamwidth for search");
    desc.add_options()("search_io_limit",
                       po::value<uint32_t>(&search_io_limit)
                           ->default_value(std::numeric_limits<_u32>::max()),
                       "Max #IOs for search");
    desc.add_options()("num_threads,T",
                       po::value<uint32_t>(&num_threads)->default_value(1),
                       "Number of threads (use 1 for instrumented search)");
    desc.add_options()("mem_L", po::value<unsigned>(&mem_L)->default_value(0),
                       "The L for in-memory navigation graph. 0 to disable.");
    desc.add_options()("use_page_search",
                       po::value<bool>(&use_page_search)->default_value(1),
                       "1 for Starling page search, 0 for DiskANN beam search");
    desc.add_options()("use_ratio",
                       po::value<float>(&use_ratio)->default_value(1.0f),
                       "Percentage of vectors in a page to search");
    desc.add_options()("disk_file_path",
                       po::value<std::string>(&disk_file_path)->required(),
                       "Path of the disk index file");
    desc.add_options()(
        "mem_index_path",
        po::value<std::string>(&mem_index_path)->default_value(""),
        "Prefix path of mem_index");
    desc.add_options()("num_queries,Q",
                       po::value<unsigned>(&num_queries)->default_value(50),
                       "Number of queries to run");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    if (vm.count("help")) {
      std::cout << desc;
      return 0;
    }
    po::notify(vm);
  } catch (const std::exception& ex) {
    std::cerr << ex.what() << '\n';
    return -1;
  }

  diskann::Metric metric;
  if (dist_fn == std::string("mips")) {
    metric = diskann::Metric::INNER_PRODUCT;
  } else if (dist_fn == std::string("l2")) {
    metric = diskann::Metric::L2;
  } else if (dist_fn == std::string("cosine")) {
    metric = diskann::Metric::COSINE;
  } else {
    std::cout << "Unsupported distance function." << std::endl;
    return -1;
  }

  try {
    if (data_type == std::string("float"))
      return run_experiment<float>(metric, index_path_prefix, mem_index_path,
                                   result_dir, query_file, disk_file_path,
                                   num_threads, K, W, search_io_limit, L, mem_L,
                                   use_page_search, use_ratio, num_queries);
    else if (data_type == std::string("int8"))
      return run_experiment<int8_t>(
          metric, index_path_prefix, mem_index_path, result_dir, query_file,
          disk_file_path, num_threads, K, W, search_io_limit, L, mem_L,
          use_page_search, use_ratio, num_queries);
    else if (data_type == std::string("uint8"))
      return run_experiment<uint8_t>(
          metric, index_path_prefix, mem_index_path, result_dir, query_file,
          disk_file_path, num_threads, K, W, search_io_limit, L, mem_L,
          use_page_search, use_ratio, num_queries);
    else {
      std::cerr << "Unsupported data type." << std::endl;
      return -1;
    }
  } catch (const std::exception& e) {
    std::cout << std::string(e.what()) << std::endl;
    std::cerr << "Experiment failed." << std::endl;
    return -1;
  }
}

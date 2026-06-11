/*
 * Copyright (c) 2026 Amazon.com, Inc. or its affiliates. All rights reserved.
 *
 * gin_rate: saturating GIN put-signal microbenchmark.
 *
 * Rank 0 keeps a fixed window of iputSignal ops in flight to every peer and
 * reposts each as it completes, with no GPU verification and no per-op barrier,
 * for a fixed wall-clock duration. Receivers spin ginProgress to reap the
 * inbound write+signal completions. Both sides drive the instrumented
 * gin_process_completions path at maximum rate, so the GIN_RATE_FINAL dump
 * reports the *saturated* per-thread completion rate (busy_pct high), answering
 * whether a single proxy progress thread can actually sustain >=1M TX + 1M RX
 * completions/sec, as opposed to the idle-bound DeepEP run.
 */

#include "config.h"

#include "functional_test.h"

#include <assert.h>
#include <deque>
#include <vector>
#include <ctime>
#include <cstdlib>

static inline ncclResult_t alloc_and_reg_buff(ncclGin_v13_t *extGin, void *collComm, size_t size,
					      int buffer_type, int value, void **buff,
					      void **mr_handle)
{
	constexpr uint64_t mrFlags = 0;
	OFINCCLCHECK(allocate_buff(buff, size, buffer_type));
	OFINCCLCHECK(initialize_buff(*buff, size, buffer_type, value));

	void *gin_handle = nullptr;
	OFINCCLCHECK(extGin->regMrSym(collComm, *buff, size, buffer_type, mrFlags, mr_handle,
				      &gin_handle));
	assert(*mr_handle != nullptr && gin_handle != nullptr);

	return ncclSuccess;
}

struct proc_handle {
	char handle[NCCL_NET_HANDLE_MAXSIZE];
};

static inline uint64_t now_ns()
{
	struct timespec t;
	clock_gettime(CLOCK_MONOTONIC, &t);
	return (uint64_t)t.tv_sec * 1000000000ULL + (uint64_t)t.tv_nsec;
}

int main(int argc, char *argv[])
{
	ncclResult_t res = ncclSuccess;
	int rank, nranks, proc_name_len, local_rank = 0;
	int buffer_type = NCCL_PTR_HOST;
	int ndev;
	int dev;

	/* Tunables (env): window of in-flight ops per peer, run duration. */
	const char *win_env = getenv("GIN_RATE_WINDOW");
	const char *dur_env = getenv("GIN_RATE_SECS");
	const int WINDOW = win_env ? atoi(win_env) : 64;
	const double RUN_SECS = dur_env ? atof(dur_env) : 5.0;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &nranks);

	std::vector<proc_handle> handles(nranks);
	std::vector<void *> handles_ptrs(nranks);

	if (nranks < 2) {
		NCCL_OFI_WARN("gin_rate needs >= 2 ranks, got %d", nranks);
		MPI_Finalize();
		return 1;
	}

	std::vector<char> all_proc_name(nranks * MPI_MAX_PROCESSOR_NAME);
	MPI_Get_processor_name(&all_proc_name[PROC_NAME_IDX(rank)], &proc_name_len);
	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, all_proc_name.data(),
		      MPI_MAX_PROCESSOR_NAME, MPI_BYTE, MPI_COMM_WORLD);

	for (int i = 0; i < nranks; i++) {
		if (!strcmp(&all_proc_name[PROC_NAME_IDX(rank)], &all_proc_name[PROC_NAME_IDX(i)])) {
			if (i < rank) {
				++local_rank;
			}
		}
	}

	CUDACHECK(cudaSetDevice(local_rank));

	set_system_page_size();
	auto *net_plugin_handle = load_netPlugin();
	auto *extNet = get_netPlugin_symbol(net_plugin_handle);
	auto *extGin = get_ginPlugin_symbol(net_plugin_handle);

	void *netCtx = nullptr;
	ncclNetCommConfig_v11_t netConfig = {};
	OFINCCLCHECK(extNet->init(&netCtx, 0, &netConfig, &functional_test_logger, nullptr));

	void *ginCtx = nullptr;
	OFINCCLCHECK(extGin->init(&ginCtx, 0, &functional_test_logger));
	OFINCCLCHECK(extGin->devices(&ndev));

	std::vector<int> test_support_gdr(ndev);
	for (dev = 0; dev < ndev; dev++) {
		ncclNetProperties_v12_t props = {};
		OFINCCLCHECK(extGin->getProperties(dev, &props));
		test_support_gdr[dev] = is_gdr_supported_nic(props.ptrSupport);
	}

	dev = local_rank % ndev;
	if (test_support_gdr[dev] == 1) {
		buffer_type = NCCL_PTR_CUDA;
	} else {
		NCCL_OFI_WARN("gin_rate: NIC lacks GDR on dev %d", dev);
		MPI_Finalize();
		return 1;
	}

	void *listenComm = nullptr;
	OFINCCLCHECK(extGin->listen(ginCtx, dev, handles[rank].handle, &listenComm));
	assert(listenComm);

	MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, handles.data(), NCCL_NET_HANDLE_MAXSIZE,
		      MPI_CHAR, MPI_COMM_WORLD);
	for (int i = 0; i < nranks; ++i) {
		handles_ptrs[i] = &(handles[i]);
	}

	void *collComm = nullptr;
	OFINCCLCHECK(extGin->connect(ginCtx, handles_ptrs.data(), nranks, rank, listenComm,
				     &collComm));
	assert(collComm != nullptr);

	ncclGinConfig_v13_t ginConfig = {};
	ginConfig.nSignals = 64;
	ginConfig.nContexts = 1;
	ginConfig.queueDepth = 64;
	ginConfig.trafficClass = -1;

	void *proxyCtx = nullptr;
	ncclNetDeviceHandle_v11_t *devHandle = nullptr;
	OFINCCLCHECK(extGin->createContext(collComm, &ginConfig, &proxyCtx, &devHandle));
	assert(proxyCtx != nullptr);

	void *put_signal_buff = nullptr;
	void *put_signal_mhandle = nullptr;
	OFINCCLCHECK(alloc_and_reg_buff(extGin, collComm, SEND_SIZE, buffer_type, 0,
					&put_signal_buff, &put_signal_mhandle));

	void *signal_buf = nullptr;
	void *signal_mhandle = nullptr;
	OFINCCLCHECK(alloc_and_reg_buff(extGin, collComm, sizeof(uint64_t), buffer_type, 0,
					&signal_buf, &signal_mhandle));

	MPI_Barrier(MPI_COMM_WORLD);

	{
		/* Symmetric all-to-all flood: every rank keeps a window of iputSignal
		 * ops in flight to every peer and reposts on completion, while the same
		 * thread reaps inbound writes/signals via ginProgress. So each progress
		 * thread handles BOTH its TX completions and all peers' RX completions
		 * concurrently — the configuration that can put 1M TX + 1M RX on one
		 * thread. */
		const int npeers = nranks - 1;
		std::deque<void *> inflight;
		uint64_t issued = 0;
		uint64_t completed = 0;
		uint64_t start = now_ns();
		uint64_t deadline = start + (uint64_t)(RUN_SECS * 1e9);
		bool draining = false;

		auto post_one = [&](int dst_rank) -> ncclResult_t {
			void *request = nullptr;
			OFINCCLCHECK(extGin->iputSignal(proxyCtx, 0, 0, put_signal_mhandle,
							SEND_SIZE, 0, put_signal_mhandle, dst_rank,
							0, signal_mhandle, 1,
							NCCL_NET_SIGNAL_OP_INC, 0, &request));
			assert(request != nullptr);
			inflight.push_back(request);
			++issued;
			return ncclSuccess;
		};

		/* Prime the window across all peers (skip self). */
		for (int w = 0; w < WINDOW; ++w) {
			for (int dst = 0; dst < nranks; ++dst) {
				if (dst == rank) {
					continue;
				}
				OFINCCLCHECK(post_one(dst));
			}
		}

		/* Round-robin refill peer, skipping self. */
		int rr = (rank + 1) % nranks;
		while (!inflight.empty()) {
			int done = 0;
			OFINCCLCHECK(extGin->test(collComm, inflight.front(), &done));
			if (done) {
				inflight.pop_front();
				++completed;
				if (!draining) {
					OFINCCLCHECK(post_one(rr));
					rr = (rr + 1) % nranks;
					if (rr == rank) {
						rr = (rr + 1) % nranks;
					}
				}
			} else {
				OFINCCLCHECK(extGin->ginProgress(proxyCtx));
			}
			if (!draining && now_ns() >= deadline) {
				draining = true;
			}
		}

		uint64_t elapsed = now_ns() - start;
		double secs = elapsed / 1e9;
		NCCL_OFI_WARN("GIN_RATE_SENDER rank=%d issued=%lu completed=%lu elapsed_s=%.3f "
			      "tx_ops_per_s=%.0f window=%d peers=%d",
			      rank, (unsigned long)issued, (unsigned long)completed, secs,
			      completed / secs, WINDOW, npeers);
	}

	/* All-to-all drain: a rank that finished sending must keep progressing its
	 * CQ so peers still sending to it can complete, until EVERY rank has stopped.
	 * A blocking barrier here would deadlock (a stopped rank would no longer
	 * service inbound writes). Use a non-blocking barrier and pump ginProgress
	 * until it completes. */
	{
		MPI_Request drain_req;
		MPI_Ibarrier(MPI_COMM_WORLD, &drain_req);
		int drain_done = 0;
		while (!drain_done) {
			OFINCCLCHECK(extGin->ginProgress(proxyCtx));
			MPI_Test(&drain_req, &drain_done, MPI_STATUS_IGNORE);
		}
	}

	OFINCCLCHECK(extGin->deregMrSym(collComm, signal_mhandle));
	signal_mhandle = nullptr;
	OFINCCLCHECK(extGin->deregMrSym(collComm, put_signal_mhandle));
	put_signal_mhandle = nullptr;

	OFINCCLCHECK(extGin->destroyContext(proxyCtx));
	proxyCtx = nullptr;
	OFINCCLCHECK(extGin->closeColl(collComm));
	collComm = nullptr;
	OFINCCLCHECK(extGin->closeListen(listenComm));
	listenComm = nullptr;

	OFINCCLCHECK(extGin->finalize(ginCtx));
	OFINCCLCHECK(extNet->finalize(netCtx));
	dlclose(net_plugin_handle);

	MPI_Barrier(MPI_COMM_WORLD);
	MPI_Finalize();

	OFINCCLCHECK(deallocate_buffer(signal_buf, buffer_type));
	signal_buf = nullptr;
	OFINCCLCHECK(deallocate_buffer(put_signal_buff, buffer_type));
	put_signal_buff = nullptr;

	NCCL_OFI_INFO(NCCL_NET, "gin_rate completed for rank %d", rank);
	return res;
}

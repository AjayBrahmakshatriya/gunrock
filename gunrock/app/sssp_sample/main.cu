#include <gunrock/gunrock.h>
//#include <gunrock/util/type_limits.cuh>
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/test_base.cuh>
#include <gunrock/app/app_base.cuh>
#include <gunrock/app/frontier.cuh>



#include <gunrock/oprtr/oprtr.cuh>
#include <gunrock/app/enactor_types.cuh>

using namespace gunrock;

typedef uint32_t VertexT;
typedef size_t SizeT;
typedef uint32_t ValueT;


typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_EDGE_VALUES | graph::HAS_CSR> GraphT;


int main(int argc, char* argv[]) {
	util::SetDevice(3);
	if (argc < 2) {
		return -1;
	}

	GraphT graph;
	
	std::string file_name(argv[1]);
	std::cout << file_name << std::endl;
	util::Parameters parameters("test pr");
	graphio::UseParameters(parameters);

	bool quiet = true;


	parameters.Parse_CommandLine(argc, argv);


	graphio::LoadGraph(parameters, graph);	



	std::cout << "Total number of nodes = " << graph.nodes << std::endl;




	util::Array1D<SizeT, ValueT> SP;
	util::Array1D<SizeT, VertexT> Labels;


	SP.Allocate(graph.nodes, util::DEVICE);
	SP.Allocate(graph.nodes, util::HOST);
	
	Labels.Allocate(graph.nodes, util::DEVICE);
	
	cudaDeviceSynchronize();

	SP.ForEach(
		[] __device__(ValueT &v) {
			v = 2147483647;
		}, graph.nodes, util::DEVICE, 0);
	SP.ForEach(
		[] __device__(ValueT &v) {
			v = 0;
		}, 1, util::DEVICE, 0);
		

	Labels.ForEach(
		[] __device__(VertexT &v) {
			v = 0;
		}, graph.nodes, util::DEVICE, 0);



	
	typedef typename app::Frontier<VertexT, SizeT, util::ARRAY_NONE, 0> FrontierT;
	typedef oprtr::OprtrParameters<GraphT, FrontierT, VertexT> OprtrParametersT;
	
	
	FrontierT frontier;
	OprtrParametersT oprtr_parameters;	
	std::vector<double> queue_factors;
	queue_factors.push_back(6.0);
	queue_factors.push_back(6.0);

	frontier.Init(2);


	oprtr_parameters.Init();
	oprtr_parameters.stream = 0;
	oprtr_parameters.frontier = &frontier;
	oprtr_parameters.cuda_props = NULL;
	oprtr_parameters.advance_mode = "";
	oprtr_parameters.filter_mode = "";

	oprtr_parameters.context = mgpu::CreateCudaDeviceAttachStream(3, 0);


	frontier.Reset();
	frontier.queue_index = 0;
	frontier.queue_reset = true;
	frontier.work_progress.Reset_();

	frontier.Allocate(graph.nodes, graph.edges, queue_factors);


	oprtr_parameters.label = 1;


	frontier.queue_length = 1;

	frontier.V_Q() -> ForEach(
		[] __device__ (VertexT &v) {
			v = 0;
		}, 1, util::DEVICE, 0);


	graph.Move(util::HOST, util::DEVICE, 0);
	auto &weights = graph.CsrT::edge_values;

	oprtr_parameters.advance_mode = "LB_CULL";

	oprtr_parameters.filter_mode = "CULL";

	auto advance_op = [SP, weights] __device__ (const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool {

		auto src_distance = SP[src];
		auto edge_weight = Load<cub::LOAD_CS>(weights + edge_id);
		
		auto new_distance = src_distance + edge_weight;

		auto old_distance = atomicMin(SP + dest, new_distance);

		if (new_distance < old_distance) 
			return true;
	

		return false;
		
	};

	

	VertexT iteration = 1;


	auto filter_op = [iteration, Labels] __device__ (const VertexT &src, VertexT &dest, const SizeT &edge_id,
            const VertexT &input_item, const SizeT &input_pos,
            SizeT &output_pos) -> bool {
			
		if (!util::isValid(dest)) return false;
		if (Labels[dest] == iteration) return false;
		Labels[dest] = iteration;
		return true;
	};	




	while(frontier.queue_length){
		oprtr::Advance<oprtr::OprtrType_V2V>(
		    graph.csr(), frontier.V_Q(), frontier.Next_V_Q(),
		    oprtr_parameters, advance_op, filter_op);

		iteration ++;
	
		/*oprtr::Filter<oprtr::OprtrType_V2V>(
			graph.csr(), frontier.V_Q(), frontier.Next_V_Q(), 
			oprtr_parameters, filter_op);
		*/
		frontier.GetQueueLength(0);
		std::cout << frontier.queue_length << std::endl;
	}

	SP.Move(util::DEVICE, util::HOST, graph.nodes, 0, 0);

	std:: cout << "DONE\n";
		

	
	for (int i = 0; i < graph.nodes; i++) 
		std::cout << i << " " << SP[i] << std::endl;

	SP.Release();
	return 0;
}

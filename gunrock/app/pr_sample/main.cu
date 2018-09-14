#include <gunrock/gunrock.h>
//#include <gunrock/util/type_limits.cuh>
#include <gunrock/graphio/graphio.cuh>
#include <gunrock/app/test_base.cuh>
#include <gunrock/app/app_base.cuh>

using namespace gunrock;

typedef uint32_t VertexT;
typedef size_t SizeT;
typedef float ValueT;


typedef typename app::TestGraph<VertexT, SizeT, ValueT, graph::HAS_COO> GraphT;


int main(int argc, char* argv[]) {
	util::SetDevice(3);
	if (argc < 2) {
		return -1;
	}

	GraphT graph;
	
	float damp = 0.85;


	std::string file_name(argv[1]);
	std::cout << file_name << std::endl;
	util::Parameters parameters("test pr");
	graphio::UseParameters(parameters);

	bool quiet = true;


	parameters.Parse_CommandLine(argc, argv);


	graphio::LoadGraph(parameters, graph);	

	//graphio::market::BuildMarketGraph(file_name, parameters, graph);
        //graph.RemoveSelfLoops_DuplicateEdges(
        //gunrock::graph::BY_ROW_ASCENDING, util::HOST, 0, true);


	//std::cout << graph.nodes << std::endl;
	//std::cout << graph.edges << std::endl;



	util::Array1D<SizeT, ValueT>old_ranks;
	util::Array1D<SizeT, ValueT>new_ranks;
	util::Array1D<SizeT, ValueT>contrib;
	util::Array1D<SizeT, SizeT>degrees;


	SizeT nodes = graph.nodes;
	float beta_score = ((((float) 1)  - damp) / nodes);

//	SizeT nodes = 2997166;
		
	old_ranks.Allocate(nodes, util::DEVICE);
	old_ranks.Allocate(nodes, util::HOST);

	new_ranks.Allocate(nodes, util::DEVICE);
	contrib.Allocate(nodes, util::DEVICE);
	degrees.Allocate(nodes, util::DEVICE);




	old_ranks.EnsureSize(nodes, util::DEVICE);
	old_ranks.EnsureSize(nodes, util::HOST);
	
	new_ranks.EnsureSize(nodes, util::DEVICE);
	contrib.EnsureSize(nodes, util::DEVICE);
	degrees.EnsureSize(nodes, util::DEVICE);
	
	ValueT initial_rank = 1.0/nodes;



	for(int i = 0; i < nodes; i++)
		old_ranks[i] = 1.0;

	//Intialization
	old_ranks.ForEach(
		[initial_rank]  __device__ (ValueT &rank) {
			rank = initial_rank;
		}, nodes, util::DEVICE, 0);
	
	new_ranks.ForEach(
		[]  __device__ (ValueT &rank) {
			rank = 0;
		}, nodes, util::DEVICE, 0);

	cudaDeviceSynchronize();
	
	
	degrees.ForEach(
		[]  __device__ (SizeT &degree) {
			degree = 0;
		}, nodes, util::DEVICE, 0);



	graph.Move(util::HOST, util::DEVICE, 0);

	oprtr::ForAll((VertexT *)NULL, 
		[graph, degrees] __device__  (VertexT *dummy, const SizeT &e) {
			VertexT src, dest;
			graph.GetEdgeSrcDest(e, src, dest);
			SizeT old_value = atomicAdd(degrees + src, 1);
		}, graph.edges, util::DEVICE, 0);

	
	for (int iteration = 0; iteration < 22; iteration++) {

		//std::cout << iteration << std::endl;
	
		oprtr::ForAll((VertexT*) NULL, 
			[graph, degrees, contrib, old_ranks] __device__ (VertexT *dummy, const SizeT &vertex) {
			contrib[vertex] = old_ranks[vertex] / degrees[vertex];
			
		}, nodes, util::DEVICE, 0);



		oprtr::ForAll((VertexT*) NULL, 
			[graph, new_ranks, contrib] __device__ (VertexT *dummy, const SizeT &e) {
				VertexT src, dest;
				graph.GetEdgeSrcDest(e, src, dest);
				atomicAdd(new_ranks + dest, contrib[src]);
				
			}, graph.edges, util::DEVICE, 0);
		
		oprtr::ForAll((VertexT*) NULL, 
			[graph, new_ranks, old_ranks, damp, beta_score] __device__ (VertexT *dummy, const SizeT &vertex) {

				float old_score = old_ranks[vertex];
				new_ranks[vertex] = (beta_score + (damp * new_ranks[vertex]));
				old_ranks[vertex] = new_ranks[vertex];
				new_ranks[vertex] = 0;

				
				
			}, nodes, util::DEVICE, 0);
	}

	old_ranks.Move(util::DEVICE, util::HOST, nodes, 0, 0);
	
	for (int i = 0; i < nodes; i++) {
		std::cout << i << " " << old_ranks[i] << std::endl;
	}	

	

	old_ranks.Release();
	new_ranks.Release();
	contrib.Release();
	degrees.Release();
	


	return 0;
}

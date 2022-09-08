#include <iostream>
#include <cstdlib>
#include "redev.h"

void rcbPtnTest(int rank, bool isRdv) {
  //dummy partition vector data: rcb partition
  const auto expectedRanks = redev::LOs({0,1,2,3});
  const auto expectedCuts = redev::Reals({0,0.5,0.75,0.25});
  auto ranks = isRdv ? expectedRanks : redev::LOs(4);
  auto cuts = isRdv ? expectedCuts : redev::Reals(4);
  const auto dim = 2;
  redev::Redev rdv(MPI_COMM_WORLD, redev::RCBPtn(dim,ranks,cuts),static_cast<redev::ProcessType>(isRdv));
  adios2::Params params{ {"Streaming", "On"}, {"OpenTimeoutSecs", "2"}};
  auto commPair = rdv.CreateDSpacesClient<redev::LO>("foo");
  if(!isRdv) {
    const auto& partition = std::get<redev::RCBPtn>(rdv.GetPartition());
    auto ptnRanks = partition.GetRanks();
    auto ptnCuts = partition.GetCuts();
    REDEV_ALWAYS_ASSERT(ptnRanks == expectedRanks);
    REDEV_ALWAYS_ASSERT(ptnCuts == expectedCuts);
  }
}

int main(int argc, char** argv) {
  int rank, nproc;
  MPI_Init(&argc, &argv);
  if(argc != 2) {
    std::cerr << "Usage: " << argv[0] << " <1=isRendezvousApp,0=isParticipant>\n";
    exit(EXIT_FAILURE);
  }
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  auto isRdv = atoi(argv[1]);
  std::cout << "comm rank " << rank << " size " << nproc << " isRdv " << isRdv << "\n";
  rcbPtnTest(rank,isRdv);
  MPI_Finalize();
  return 0;
}

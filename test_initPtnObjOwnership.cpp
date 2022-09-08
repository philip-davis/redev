#include <iostream>
#include "redev.h"

const std::string timeout="8";

auto makeRedev(int dim, redev::LOs& ranks, redev::Reals& cuts, bool isRendezvous) {
  auto ptn = redev::RCBPtn(dim,ranks,cuts);
  return redev::Redev(MPI_COMM_WORLD,ptn,static_cast<redev::ProcessType>(isRendezvous));
}

void client() {
  const auto dim = 1;
  auto ranks = redev::LOs(1);
  auto cuts = redev::Reals(1);
  const bool isRendezvous = false;
  auto rdv = makeRedev(dim, ranks, cuts, isRendezvous);
  auto commPair = rdv.CreateDSpacesClient<redev::LO>("foo");
}

void server() {
  const auto dim = 1;
  auto ranks = redev::LOs({0});
  auto cuts = redev::Reals({0});
  const bool isRendezvous=true;
  auto rdv = makeRedev(dim, ranks, cuts, isRendezvous);
  auto commPair = rdv.CreateDSpacesClient<redev::LO>("foo");
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int nproc;
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  REDEV_ALWAYS_ASSERT(nproc == 1);
  REDEV_ALWAYS_ASSERT(argc == 2);
  const int isRdv = atoi(argv[1]);
  std::cout << "isRdv " << isRdv << "\n";
  REDEV_ALWAYS_ASSERT(isRdv == 1 || isRdv == 0);
  if(isRdv)
    server();
  else
    client();
  std::cout << "done\n";
  MPI_Finalize();
  return 0;
}

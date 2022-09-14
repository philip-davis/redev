#include <redev.h>
#include <cassert>
#include <span>
#include "redev_git_version.h"
#include "redev.h"
#include "redev_profile.h"
#include "redev_scan.h"
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds
#include <string>         // std::stoi
#include <cstring>

namespace {
  //Wait for the file to be created by the writer.
  //Assuming that if 'Streaming' and 'OpenTimeoutSecs' are set then we are in
  //BP4 mode.  SST blocks on Open by default.
  void waitForEngineCreation(adios2::IO& io) {
    REDEV_FUNCTION_TIMER;
    auto params = io.Parameters();
    bool isStreaming = params.count("Streaming") &&
                       redev::isSameCaseInsensitive(params["Streaming"], "ON");
    bool timeoutSet = params.count("OpenTimeoutSecs") && std::stoi(params["OpenTimeoutSecs"]) > 0;
    bool isSST = redev::isSameCaseInsensitive(io.EngineType(), "SST");
    if( (isStreaming && timeoutSet) || isSST ) return;
    std::this_thread::sleep_for(std::chrono::seconds(2));
  }

}

namespace redev {

  //TODO consider moving the ClassPtn source to another file
  ClassPtn::ClassPtn() {}

  ClassPtn::ClassPtn(MPI_Comm comm, const redev::LOs& ranks_, const ModelEntVec& ents) {
    assert(ranks_.size() == ents.size());
    if( ! ModelEntDimsValid(ents) ) exit(EXIT_FAILURE);
    for(auto i=0; i<ranks_.size(); i++) {
      modelEntToRank[ents[i]] = ranks_[i];
    }
    Gather(comm);
    metaStep = 0;
  }

  void ClassPtn::Gather(MPI_Comm comm, int root) {
    REDEV_FUNCTION_TIMER;
    int rank, commSize;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &commSize);
    auto degree = !rank ? redev::LOs(commSize+1) : redev::LOs();
    auto serialized = SerializeModelEntsAndRanks();
    int len = static_cast<int>(serialized.size());
    MPI_Gather(&len,1,MPI_INT,degree.data(),1,MPI_INT,root,comm);
    if(root==rank) {
      auto offset = redev::LOs(commSize+1);
      redev::exclusive_scan(degree.begin(), degree.end(), offset.begin(), redev::LO(0));
      auto allSerialized = redev::LOs(offset.back());
      MPI_Gatherv(serialized.data(), len, MPI_INT, allSerialized.data(),
          degree.data(), offset.data(), MPI_INT, root, MPI_COMM_WORLD);
      modelEntToRank = DeserializeModelEntsAndRanks(allSerialized);
    } else {
      MPI_Gatherv(serialized.data(), len, MPI_INT, NULL, NULL, NULL, MPI_INT, root, MPI_COMM_WORLD);
    }
  }

  bool ClassPtn::ModelEntDimsValid(const ClassPtn::ModelEntVec& ents) const {
    auto badDim = [](ModelEnt e){ return (e.first<0 || e.first>3); };
    auto res = std::find_if(begin(ents), end(ents), badDim);
    if (res != std::end(ents)) {
      std::stringstream ss;
      ss << "ERROR: a ModelEnt contains an invalid dimension: "
         << res->first << "... exiting\n";
      std::cerr << ss.str();
    }
    return (res == std::end(ents));
  }

  redev::LO ClassPtn::GetRank(ModelEnt ent) const {
    REDEV_FUNCTION_TIMER;
    REDEV_ALWAYS_ASSERT(ent.first>=0 && ent.first <=3); //check for valid dimension
    assert(modelEntToRank.size());
    assert(modelEntToRank.count(ent));
    return modelEntToRank.at(ent);
  }

  redev::LOs ClassPtn::GetRanks() const {
    REDEV_FUNCTION_TIMER;
    redev::LOs ranks(modelEntToRank.size());
    int i=0;
    for(const auto& iter : modelEntToRank) {
      ranks[i++]=iter.second;
    }
    return ranks;
  }

  ClassPtn::ModelEntVec ClassPtn::GetModelEnts() const {
    REDEV_FUNCTION_TIMER;
    ModelEntVec ents(modelEntToRank.size());
    int i=0;
    for(const auto& iter : modelEntToRank) {
      ents[i++]=iter.first;
    }
    return ents;
  }

  redev::LOs ClassPtn::SerializeModelEntsAndRanks() const {
    REDEV_FUNCTION_TIMER;
    const auto stride = 3;
    const auto numEnts = modelEntToRank.size();
    redev::LOs entsAndRanks;
    entsAndRanks.reserve(numEnts*stride);
    for(const auto& iter : modelEntToRank) {
      auto ent = iter.first;
      entsAndRanks.push_back(ent.first);   //dim
      entsAndRanks.push_back(ent.second);  //id
      entsAndRanks.push_back(iter.second); //rank
    }
    REDEV_ALWAYS_ASSERT(entsAndRanks.size()==numEnts*stride);
    return entsAndRanks;
  }

  ClassPtn::ModelEntToRank ClassPtn::DeserializeModelEntsAndRanks(const std::span<redev::LO> serialized) const {
    REDEV_FUNCTION_TIMER;
    const auto stride = 3;
    REDEV_ALWAYS_ASSERT(serialized.size()%stride==0);
    ModelEntToRank me2r;
    for(size_t i=0; i<serialized.size(); i+=stride) {
      const auto dim=serialized[i];
      REDEV_ALWAYS_ASSERT(dim>=0 && dim<=3);
      const auto id=serialized[i+1];
      const auto rank=serialized[i+2];
      ModelEnt ent(dim,id);
      const auto hasEnt = me2r.count(ent);
      if( (hasEnt && rank < me2r[ent]) || !hasEnt ) {
        me2r[ModelEnt(dim,id)] = rank;
      }
    }
    return me2r;
  }

  void ClassPtn::Write(dspaces_client_t dsp, std::string_view name) {
    REDEV_FUNCTION_TIMER;
    char var_name[256];
    uint64_t lb, ub;
    auto serialized = SerializeModelEntsAndRanks();
    uint64_t len = serialized.size();
    sprintf(var_name, "%s_%s", name.data(), entsAndRanksVarName.c_str());
    dspaces_put_meta(dsp, var_name, metaStep++, serialized.data(), sizeof(decltype(serialized)::value_type) * len);
  }

  void ClassPtn::Read(dspaces_client_t dsp, std::string_view name) {
    REDEV_FUNCTION_TIMER;
    char var_name[256];
    int step;
    redev::LO *entAndRanks;
    unsigned int size, len;
    sprintf(var_name, "%s_%s", name.data(), entsAndRanksVarName.c_str());
    dspaces_get_meta(dsp, var_name, META_MODE_LAST, -1, &step, (void **)&entAndRanks, &size);
    assert(size && size % sizeof(redev::LO) == 0);
    len = size / sizeof(redev::LO);
    modelEntToRank = DeserializeModelEntsAndRanks(std::span<redev::LO>(entAndRanks, len));
    free(entAndRanks);
  }

  void ClassPtn::Broadcast(MPI_Comm comm, int root) {
    REDEV_FUNCTION_TIMER;
    int rank;
    MPI_Comm_rank(comm, &rank);
    auto serialized = SerializeModelEntsAndRanks();
    int len = static_cast<int>(serialized.size());
    redev::Broadcast(&len, 1, root, comm);
    if(root != rank) {
      serialized.resize(len);
    }
    redev::Broadcast(serialized.data(), serialized.size(), root, comm);
    if(root != rank) {
      modelEntToRank = DeserializeModelEntsAndRanks(serialized);
    }
  }


  //TODO consider moving the RCBPtn source to another file
  RCBPtn::RCBPtn() {}

  RCBPtn::RCBPtn(redev::LO dim_)
    : dim(dim_) {
    assert(dim>0 && dim<=3);
    metaStep = 0;
  }

  RCBPtn::RCBPtn(redev::LO dim_, std::vector<int>& ranks_, std::vector<double>& cuts_)
    : dim(dim_), ranks(ranks_), cuts(cuts_) {
    assert(dim>0 && dim<=3);
  }

  redev::LO RCBPtn::GetRank(std::array<redev::Real,3>& pt) const { //TODO better name?
    REDEV_FUNCTION_TIMER;
    assert(ranks.size() && cuts.size());
    assert(dim>0 && dim<=3);
    const auto len = cuts.size();
    const auto levels = std::log2(len);
    auto lvl = 0;
    auto idx = 1;
    auto d = 0;
    while(lvl < levels) {
      if(pt[d]<cuts.at(idx))
        idx = idx*2;
      else
        idx = idx*2+1;
      ++lvl;
      d = (d + 1) % dim;
    }
    auto rankIdx = idx-std::pow(2,lvl);
    assert(rankIdx < ranks.size());
    return ranks.at(rankIdx);
  }

  std::vector<redev::LO> RCBPtn::GetRanks() const {
    REDEV_FUNCTION_TIMER;
    return ranks;
  }

  std::vector<redev::Real> RCBPtn::GetCuts() const {
    REDEV_FUNCTION_TIMER;
    return cuts;
  }

  void RCBPtn::Write(dspaces_client_t dsp, std::string_view name) {
    REDEV_FUNCTION_TIMER;
    char var_name[256];
    uint64_t lb, ub;
    uint64_t len = ranks.size();
    if(!len) return; //don't attempt zero length write
    assert(len==cuts.size());
    sprintf(var_name, "%s_%s", name.data(), ranksVarName.c_str());
    dspaces_put_meta(dsp, var_name, 0, ranks.data(), sizeof(decltype(ranks)::value_type) * len);
    sprintf(var_name, "%s_%s", name.data(), cutsVarName.c_str());
    dspaces_put_meta(dsp, var_name, 0, cuts.data(), sizeof(decltype(cuts)::value_type) * len);
  }

  void RCBPtn::Read(dspaces_client_t dsp, std::string_view name) {
    REDEV_FUNCTION_TIMER;
    char var_name[256];
    int step;
    redev::LO *ranksBuf;
    redev::Real *cutsBuf;
    unsigned int size, len;

    sprintf(var_name,  "%s_%s", name.data(), ranksVarName.c_str());
    dspaces_get_meta(dsp, var_name, META_MODE_LAST, -1, &step, (void **)&ranksBuf, &size);
    assert(size && size % sizeof(redev::LO) == 0);
    len = size / sizeof(redev::LO);
    ranks.assign(ranksBuf, ranksBuf + len);
    free(ranksBuf);

    sprintf(var_name,  "%s_%s", name.data(), cutsVarName.c_str());
    dspaces_get_meta(dsp, var_name, META_MODE_LAST, -1, &step, (void **)&cutsBuf, &size);
    assert(size && size % sizeof(redev::Real) == 0);
    len = size / sizeof(redev::Real);
    cuts.assign(cutsBuf, cutsBuf + len);
    free(cutsBuf);
  }

  void RCBPtn::Broadcast(MPI_Comm comm, int root) {
    REDEV_FUNCTION_TIMER;
    int rank;
    MPI_Comm_rank(comm, &rank);
    int count = ranks.size();
    redev::Broadcast(&count, 1, root, comm);
    if(root != rank) {
      ranks.resize(count);
      cuts.resize(count);
    }
    redev::Broadcast(ranks.data(), ranks.size(), root, comm);
    redev::Broadcast(cuts.data(), cuts.size(), root, comm);
  }

  // BP4 support
  // - with a rendezvous + non-rendezvous application pair
  // - with only a rendezvous application for debugging/testing
  // - in streaming and non-streaming modes; non-streaming requires 'waitForEngineCreation'
  void Redev::openEnginesBP4(bool noClients,
      std::string s2cName, std::string c2sName,
      adios2::IO& s2cIO, adios2::IO& c2sIO,
      adios2::Engine& s2cEngine, adios2::Engine& c2sEngine) {
    REDEV_FUNCTION_TIMER;
    //create the engine writers at the same time - BP4 does not wait for the readers (SST does)
    if(processType == ProcessType::Server) {
      s2cEngine = s2cIO.Open(s2cName, adios2::Mode::Write);
      assert(s2cEngine);
    } else {
      c2sEngine = c2sIO.Open(c2sName, adios2::Mode::Write);
      assert(c2sEngine);
    }
    waitForEngineCreation(s2cIO);
    waitForEngineCreation(c2sIO);
    //create engines for reading
    if(processType == ProcessType::Server) {
      if(noClients==false) { //support unit testing
        c2sEngine = c2sIO.Open(c2sName, adios2::Mode::Read);
        assert(c2sEngine);
      }
    } else {
      s2cEngine = s2cIO.Open(s2cName, adios2::Mode::Read);
      assert(s2cEngine);
    }
  }

  // SST support
  // - with a rendezvous + non-rendezvous application pair
  // - with only a rendezvous application for debugging/testing
  void Redev::openEnginesSST(bool noClients,
      std::string s2cName, std::string c2sName,
      adios2::IO& s2cIO, adios2::IO& c2sIO,
      adios2::Engine& s2cEngine, adios2::Engine& c2sEngine) {
    REDEV_FUNCTION_TIMER;
    //create one engine's reader and writer pair at a time - SST blocks on open(read)
    if(processType==ProcessType::Server) {
      s2cEngine = s2cIO.Open(s2cName, adios2::Mode::Write);
    } else {
      s2cEngine = s2cIO.Open(s2cName, adios2::Mode::Read);
    }
    assert(s2cEngine);
    if(processType==ProcessType::Server) {
      if(noClients==false) { //support unit testing
        c2sEngine = c2sIO.Open(c2sName, adios2::Mode::Read);
        assert(c2sEngine);
      }
    } else {
      c2sEngine = c2sIO.Open(c2sName, adios2::Mode::Write);
      assert(c2sEngine);
    }
  }

  Redev::Redev(MPI_Comm comm, Partition ptn, ProcessType processType, bool noClients)
    : comm(comm), ptn(ptn), processType(processType), noClients(noClients) {
    REDEV_FUNCTION_TIMER;
    int isInitialized = 0;
    MPI_Initialized(&isInitialized);
    REDEV_ALWAYS_ASSERT(isInitialized);
    MPI_Comm_rank(comm, &rank); //set member var
    if(processType == ProcessType::Server) {
        char *conn_str = std::getenv("DSP_SRV_CONN");
        dspaces_server_init(conn_str ? conn_str : "sockets", comm, "dataspaces.conf", &dsp_srv);
    }
    dspaces_init_mpi(comm, &dsp);
  }

  Redev::~Redev() {
    if(rank == 0) {
        dspaces_kill(dsp);
    }
    dspaces_fini(dsp);
    if(processType == ProcessType::Server) {
        dspaces_server_fini(dsp_srv);
    }
    MPI_Barrier(comm);
  }

  void Redev::Setup(std::string_view name) {
    REDEV_FUNCTION_TIMER;
    CheckVersion(name);
    //rendezvous app rank 0 writes partition info and other apps read
    if(!rank) {
      if(processType==ProcessType::Server) {
        std::visit([&](auto&& partition){partition.Write(dsp, name);} ,ptn);
      }
      else {
        std::visit([&](auto&& partition){partition.Read(dsp, name);} ,ptn);
      }
    }
    std::visit([&](auto&& partition){partition.Broadcast(comm);} ,ptn);

  }

  /*
   * return the number of processes in the client's MPI communicator
   */
  redev::LO Redev::GetClientCommSize(std::string_view name) {
    REDEV_FUNCTION_TIMER;
    char var_name[256];
    int commSize, step;
    unsigned int size;
    int *sizeMeta;
    MPI_Comm_size(comm, &commSize);
    const auto varName = "redev client communicator size";
    redev::LO clientCommSz = 0;
    sprintf(var_name, "%s_%s", name.data(), varName);
    if(processType == ProcessType::Client) {
      if(!rank)
        dspaces_put_meta(dsp, var_name, 0, &commSize, sizeof(commSize));
    } else {
      if(!rank) {
        dspaces_get_meta(dsp, var_name, META_MODE_NEXT, -1, &step, (void **)&sizeMeta, &size);
        assert(step == 0 && size == sizeof(commSize));
        clientCommSz = *sizeMeta;
        free(sizeMeta);
      }
    }
    if(processType==ProcessType::Server)
      redev::Broadcast(&clientCommSz,1,0,comm);
    return clientCommSz;
  }

  /*
   * return the number of processes in the server's MPI communicator
   */
  redev::LO Redev::GetServerCommSize(std::string_view name) {
    REDEV_FUNCTION_TIMER;
    char var_name[256];
    int commSize, step;
    unsigned int size;
    int *sizeMeta;
    MPI_Comm_size(comm, &commSize);
    const auto varName = "redev server communicator size";
    redev::LO serverCommSz = 0;
    sprintf(var_name, "%s_%s", name.data(), varName);
    if(processType==ProcessType::Server) {
      if(!rank)
          dspaces_put_meta(dsp, var_name, 0, &commSize, sizeof(commSize));
    } else {
      if(!rank) {
        dspaces_get_meta(dsp, var_name, META_MODE_NEXT, -1, &step, (void **)&sizeMeta, &size);
        assert(step == 0 && size == sizeof(commSize));
        serverCommSz = *sizeMeta;
        free(sizeMeta);
      }
    }
    if(processType == ProcessType::Client)
      redev::Broadcast(&serverCommSz,1,0,comm);
    return serverCommSz;
  }

  void Redev::CheckVersion(std::string_view name) {
    REDEV_FUNCTION_TIMER;
    char *inHash;
    int step;
    unsigned int size;

    const auto hashVarName = "redev git hash";
    //rendezvous app writes the version it has and other apps read
    if(processType==ProcessType::Server) {
      if(!rank) {
          dspaces_put_meta(dsp, hashVarName, 0, (void *)redevGitHash, strlen(redevGitHash) + 1);
      }
    }
    else {
      if(!rank) {
        dspaces_get_meta(dsp, hashVarName, META_MODE_NEXT, -1, &step, (void **)&inHash, &size);
        assert(size);
        std::cout << "inHash " << inHash << "\n";
        REDEV_ALWAYS_ASSERT(std::string(inHash) == redevGitHash);
        free(inHash);
      }
    }
  }
  ProcessType Redev::GetProcessType() const { return processType; }
  const Partition &Redev::GetPartition() const {return ptn;}
  }

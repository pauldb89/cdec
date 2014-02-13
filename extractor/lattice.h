#pragma once

#include <memory>
#include <string>
#include <vector>

using namespace std;

namespace extractor {

class Vocabulary;

typedef vector<vector<pair<int, int>>> DAG;

class Lattice {
 public:
  Lattice(const string& input,
          shared_ptr<Vocabulary> vocabulary,
          bool sentence);

  size_t GetSize() const;

  vector<pair<int, int>> GetTransitions(int node) const;

  vector<pair<int, int>> GetExtensions(int node, int low, int high) const;

 private:
  void ParseSentence(const string& input, shared_ptr<Vocabulary> vocabulary);

  void ParseLattice(const string& input, shared_ptr<Vocabulary> vocabulary);

  void RemoveEpsilonTransitions(DAG egraph, const int epsilon);

  void ComputeDistances();

  DAG graph;
  vector<vector<int>> distance;
};

} // namespace extractor

#include "lattice.h"

#include <sstream>

#include "vocabulary.h"

namespace extractor {

Lattice::Lattice(const string& input,
                 shared_ptr<Vocabulary> vocabulary,
                 bool sentence) {
  if (sentence) {
    ParseSentence(input, vocabulary);
  } else {
    ParseLattice(input, vocabulary);
  }
  ComputeDistances();
}

void Lattice::ParseSentence(
    const string& sentence, shared_ptr<Vocabulary> vocabulary) {
  vector<string> words;
  istringstream buffer(sentence);
  copy(istream_iterator<string>(buffer),
       istream_iterator<string>(),
       back_inserter(words));

  graph.resize(words.size() + 1);
  for (size_t i = 0; i < words.size(); ++i) {
    int word_id = vocabulary->GetTerminalIndex(words[i]);
    graph[i] = vector<pair<int, int>>(1, make_pair(i + 1, word_id));
  }
}

void Lattice::ParseLattice(
    const string& lattice, shared_ptr<Vocabulary> vocabulary) {
  // This code makes the following assumptions:
  // 1. Each node except the last in the topological order has at least one
  //    (possibly epsilon) outgoing edge.
  // 2. The lattice has at least two nodes.
  // 3. Each edge label has at most 50 characters.
  // See unit tests for examples of parsable inputs.

  DAG egraph;
  int current_node = 0;
  char last_char;
  istringstream iss(lattice);
  assert(iss.get() == '(');
  while ((last_char = iss.get()) == '(') {
    egraph.resize(egraph.size() + 1);

    while ((last_char = iss.get()) == '(') {
      char current_char;
      string word;
      double prob;
      int node_delta;
      assert(iss.get() == '\'');
      // Do not stop at escaped apostrophes.
      while ((current_char = iss.get()) != '\'') {
        word.append(1, current_char != '\\' ? current_char : iss.get());
      }
      assert(iss.get() == ',');
      assert(iss >> prob);
      assert(iss.get() == ',');
      assert(iss >> node_delta);
      assert(iss.get() == ')');
      assert(iss.get() == ',');

      assert(node_delta > 0);
      int word_id = vocabulary->GetTerminalIndex(string(word));
      int next_node = current_node + node_delta;
      egraph[current_node].push_back(make_pair(next_node, word_id));
    }

    assert(last_char == ')');
    assert(iss.get() == ',');
    ++current_node;
  }
  assert(last_char == ')');

  egraph.resize(egraph.size() + 1);
  RemoveEpsilonTransitions(egraph, vocabulary->GetTerminalIndex("*EPS*"));
}

void Lattice::RemoveEpsilonTransitions(DAG egraph, const int epsilon) {
  // Transform any non-epsilon edge followed by an epsilon edge into a single
  // non-epsilon edge. Leave epsilon edges in for the second step.
  graph = DAG(egraph.size());
  DAG rev_graph(graph.size());
  for (size_t node = 0; node < egraph.size(); ++node) {
    for (const auto& edge: egraph[node]) {
      if (edge.second == epsilon) {
        for (const auto& rev_edge: rev_graph[node]) {
          if (rev_edge.second != epsilon) {
            graph[rev_edge.first].push_back(make_pair(
                edge.first, rev_edge.second));
            rev_graph[edge.first].push_back(rev_edge);
          }
        }
      }
      graph[node].push_back(edge);
      rev_graph[edge.first].push_back(make_pair(node, edge.second));
    }
  }

  // Transform any epsilon edge followed by an non-epsilon edge into a single
  // non-epsilon edge. Remove epsilon edges.
  egraph = graph;
  graph = DAG(egraph.size());
  for (int node = egraph.size() - 1; node >= 0; --node) {
    for (const auto& rev_edge: rev_graph[node]) {
      if (rev_edge.second == epsilon) {
        for (const auto& edge: graph[node]) {
          graph[rev_edge.first].push_back(edge);
        }
      } else {
        graph[rev_edge.first].push_back(make_pair(node, rev_edge.second));
      }
    }
  }
}

void Lattice::ComputeDistances() {
  distance.resize(graph.size(), vector<int>(graph.size(), graph.size()));
  for (size_t start_node = 0; start_node < graph.size(); ++start_node) {
    distance[start_node][start_node] = 0;
    for (size_t node = start_node; node < graph.size(); ++node) {
      for (const auto& edge: graph[node]) {
        distance[start_node][edge.first] = min(
            distance[start_node][edge.first], distance[start_node][node] + 1);
      }
    }
  }
}

size_t Lattice::GetSize() const {
  return graph.size();
}

vector<pair<int, int>> Lattice::GetTransitions(int node) const {
  return graph[node];
}

vector<pair<int, int>> Lattice::GetExtensions(
    int start_node, int low, int high) const {
  vector<pair<int, int>> extensions;
  for (size_t node = start_node + 1; node < graph.size(); ++node) {
    if (distance[start_node][node] >= max(low, 1) &&
        distance[start_node][node] <= min(high, (int) graph.size() - 1)) {
      extensions.push_back(make_pair(node, distance[start_node][node]));
    }
  }
  return extensions;
}

} // namespace extractor

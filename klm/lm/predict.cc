#include <fstream>
#include <iostream>
#include <vector>

#include <boost/program_options.hpp>

#include "lm/model.hh"
#include "lm/enumerate_vocab.hh"
#include "lm/vocab.hh"

using namespace boost::program_options;
using namespace lm;
using namespace lm::ngram;
using namespace std;

int main(int argc, char** argv) {
  options_description desc("Command line options");
  desc.add_options()
      ("help,h", "Print help message")
      ("model,m", value<string>()->required(),
       "File containing the model in binary format")
      ("contexts,c", value<string>()->required(),
       "File containing the contexts to be processed");

  variables_map vm;
  store(parse_command_line(argc, argv, desc), vm);
  if (vm.count("help")) {
    cout << desc << endl;
    return 0;
  }

  notify(vm);

  Config config;
  vector<WordIndex> v;
  VocabularyMapper* vmapper = new VocabularyMapper();
  config.enumerate_vocab = vmapper;
  ProbingModel model(vm["model"].as<string>().c_str(), config);
  vector<string> words = vmapper->getWords();

  string line;
  ifstream in(vm["contexts"].as<string>());
  while (getline(in, line)) {
    string word;
    istringstream iss(line);
    vector<WordIndex> context;
    vector<string> context_words;
    while (iss >> word) {
      context.push_back(model.GetVocabulary().Index(word));
      context_words.push_back(word);
    }

    ProbingModel::State state = model.BeginSentenceState(), out;
    for (size_t i = 0; i < context.size(); ++i) {
      model.FullScore(state, context[i], out);
      state = out;
    }

    vector<pair<double, string>> outcomes;
    for (const string& word: words) {
      outcomes.push_back(make_pair(
          exp(log(10) * model.FullScore(state, model.GetVocabulary().Index(word), out).prob), word));
    }

    sort(outcomes.begin(), outcomes.end());

    double sum = 0;
    for (const auto& outcome: outcomes) {
      for (const string& context_word: context_words) {
        cout << context_word << " ";
      }
      cout << outcome.second << " " << outcome.first << endl;
      sum += outcome.first;
    }

    cout << "sum: " << sum << endl;
    assert(fabs(1 - sum) < 1e-5);
    cout << "====================" << endl;
  }

  return 0;
}

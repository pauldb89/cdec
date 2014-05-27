#include "ff_lbl.h"

#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/make_shared.hpp>
#include <boost/program_options.hpp>

#include "tdict.h"
#include "lm/config.hh"

#include <algorithm>

#define kORDER 5

using namespace boost::program_options;
using namespace lm;
using namespace lm::ngram;
using namespace std;

void ParseOptions(const string& input, string& filename, string& featname) {
  options_description options("LBL language model options");
  options.add_options()
      ("file,f", value<string>()->required(),
          "File containing serialized language model")
      ("name,n", value<string>()->default_value("LanguageModel"),
          "Feature name");

  variables_map vm;
  vector<string> args;
  boost::split(args, input, boost::is_any_of(" "));
  store(command_line_parser(args).options(options).run(), vm);
  notify(vm);

  filename = vm["file"].as<string>();
  featname = vm["name"].as<string>();
}


void CdecMapper::Add(WordIndex word_id, const StringPiece& word) {
  int cdec_id = TD::Convert(word.as_string());
  if (cdec_id >= cdec2lbl.size()) {
    cdec2lbl.resize(cdec_id + 1);
  }
  cdec2lbl[cdec_id] = word_id;

  if (word_id >= words.size()) {
    words.resize(word_id + 1);
  }
  words[word_id] = word.as_string();
}

WordIndex CdecMapper::MapWord(WordID cdec_word_id) const {
  if (cdec_word_id < cdec2lbl.size()) {
    return cdec2lbl[cdec_word_id];
  } else {
    return 0;
  }
}

size_t CdecMapper::size() const {
  return cdec2lbl.size();
}

StringPiece CdecMapper::getWord(WordIndex word_id) const {
  if (word_id == -1) {
    return "<none>";
  }
  if (word_id == words.size()) {
    return "<star>";
  }
  return words[word_id];
}


SimplePair::SimplePair() : first(), second() {}

SimplePair::SimplePair(double x, double y) : first(x), second(y) {}

SimplePair& SimplePair::operator+=(const SimplePair& o) {
  first += o.first;
  second += o.second;
  return *this;
}


FF_LBLLM::FF_LBLLM(const string& filename, const string& featname) {
  Config config;
  config.enumerate_vocab = &mapper;
  model = boost::make_shared<ProbingModel>(filename.c_str(), config);
  fid = FD::Convert(featname);
  fidOOV = FD::Convert(featname + "_OOV");

  kNONE = -1;
  kUNKNOWN = model->GetVocabulary().Index("<unk>");
  kSTAR = model->GetVocabulary().Bound();
  kSTART = model->GetVocabulary().Index("<s>");
  kSTOP = model->GetVocabulary().Index("</s>");

  stateOffset = ReserveStateSize() - 1;
  FeatureFunction::SetStateSize(ReserveStateSize());
}

void FF_LBLLM::TraversalFeaturesImpl(
    const SentenceMetadata&, const HG::Edge& edge,
    const vector<const void*>& ant_states, SparseVector<double>* features,
    SparseVector<double>* estimated_features, void* state) const {
  if (debug) {
    // cout << "end debug" << endl;
  }
  debug = false;
  SimplePair ft = LookupWords(*edge.rule_, ant_states, state);
  if (ft.first) {
    features->set_value(fid, ft.first);
  }
  if (ft.second) {
    features->set_value(fidOOV, ft.second);
  }
  if (debug) {
    // cout << ft.first << " " << ft.second << endl;
  }
  SimplePair ft2 = EstimateProb(state);
  if (ft2.first) {
    estimated_features->set_value(fid, ft2.first);
  }
  if (ft2.second) {
    estimated_features->set_value(fidOOV, ft2.second);
  }

  if (debug) {
    // cout << ft2.first << " " << ft2.second << endl;
  }
}

void PrintBuffer(
    vector<WordIndex> buffer, const CdecMapper& mapper, bool rev = false) {
  if (rev) {
    reverse(buffer.begin(), buffer.end());
  }
  for (int i = buffer.size() - 1; i >= 0; --i) {
    cout << mapper.getWord(buffer[i]) << " ";
  }
  cout << endl;
}

template<class Model>
int ReduceLeft(const vector<WordIndex>& left, const boost::shared_ptr<Model>& model) {
  // Reduce left.
  for (int start_left = left.size() - 1; start_left >= 0; --start_left) {
    typename Model::Node node;
    bool should_stop;
    uint64_t ignore;
    model->search_.LookupUnigram(left[start_left], node, should_stop, ignore);
    should_stop = false;
    for (int k = start_left - 1; k >= 0; --k) {
      model->search_.LookupMiddle(start_left - k - 1, left[k], node, should_stop, ignore);
      if (should_stop) {
        break;
      }
    }

    if (!should_stop) {
      return start_left + 1;
    }
  }
  assert(false);
  return 0;
}

template<class Model>
int ReduceRight(const vector<WordIndex>& right, const boost::shared_ptr<Model>& model) {
  typename Model::Node node;
  bool should_stop;
  uint64_t ignore;
  model->search_.LookupUnigram(right.back(), node, should_stop, ignore);
  should_stop = false;
  for (int k = right.size() - 2; k >= 0; --k) {
    model->search_.LookupMiddle(right.size() - k - 2, right[k], node, should_stop, ignore);
    if (should_stop) {
      return k + 1;
    }
  }

  return 0;
}

void ExtendState(const vector<WordIndex>& values, WordIndex* state, int& len) {
  for (WordIndex value: values) {
    state[len++] = value;
  }
}

SimplePair FF_LBLLM::LookupWords(
    const TRule& rule, const vector<const void*>& ant_states,
    void* vstate) const {
  int len = rule.ELength() - rule.Arity();
  for (unsigned i = 0; i < ant_states.size(); ++i) {
    len += StateSize(ant_states[i]);
  }
  buffer.resize(len + 1);
  buffer[len] = kNONE;
  int i = len - 1;
  const vector<WordID>& e = rule.e();
  for (unsigned j = 0; j < e.size(); ++j) {
    if (e[j] < 1) {
      const WordIndex* astate =
          reinterpret_cast<const WordIndex*>(ant_states[-e[j]]);
      int slen = StateSize(astate);
      for (int k = 0; k < slen; ++k)
        buffer[i--] = astate[k];
      cout << "[X, " << -e[j] << "] ";
    } else {
      buffer[i--] = mapper.MapWord(e[j]);
      cout << TD::Convert(e[j]) << " ";
    }
  }

  vector<WordID> expected = {892, 5, 6, 0};
  if (e == expected) {
    debug = true;
    // cout << "begin debug" << endl;
    // PrintBuffer(buffer, mapper);
  }

  SimplePair sum;
  vector<WordIndex> left;
  i = len - 1;
  int edge = len;

  while (i >= 0) {
    if (buffer[i] == kSTAR) {
      edge = i;
    } else if (edge-i >= kORDER) {
      //cerr << "X: ";
      sum += LookupProbForBufferContents(i);
    } else if (edge == len) {
      left.push_back(buffer[i]);
    }
    --i;
  }

  // Extend next state to the left.
  int next_len = 0;
  WordIndex* next_state = reinterpret_cast<WordIndex*>(vstate);
  for (int word_id: left) {
    next_state[next_len++] = word_id;
  }

  if (edge != len || len >= kORDER) {
    cout << "split" << endl;
    cout << "red left: " << ReduceLeft<Model>(left, model) << endl;
    left.resize(ReduceLeft<Model>(left, model));
    ExtendState(left, next_state, next_len);

    next_state[next_len++] = kSTAR;
    if (kORDER-1 < edge) edge = kORDER-1;

    vector<WordIndex> right;
    for (int i = edge-1; i >= 0; --i) {
      right.push_back(buffer[i]);
    }

    cout << "red right: " << ReduceRight<Model>(right, model) << endl;
    right.erase(right.begin(), right.begin() + ReduceRight<Model>(right, model));

    ExtendState(right, next_state, next_len);
  } else {
    int reduce_left = ReduceLeft<Model>(left, model);
    int reduce_right = ReduceRight<Model>(left, model);
    cout << "joint" << endl;
    cout << "red left: " << reduce_left << endl;
    cout << "red right: " << reduce_right << endl;
    left.erase(left.begin() + reduce_left, left.begin() + reduce_right);
    ExtendState(left, next_state, next_len);
  }

  vector<WordIndex> next(next_len);
  for (int i = 0; i < next_len; ++i) {
    next[i] = next_state[i];
  }
  cout << "final state: ";
  PrintBuffer(next, mapper);

  SetStateSize(next_len, vstate);
  assert(StateSize(vstate) == next_len);

  return sum;
}

void FF_LBLLM::FinalTraversalFeatures(
    const void* ant_state, SparseVector<double>* features) const {
  const SimplePair ft = FinalTraversalCost(ant_state);
  if (ft.first) {
    features->set_value(fid, ft.first);
  }
  if (ft.second) {
    features->set_value(fidOOV, ft.second);
  }
}

SimplePair FF_LBLLM::FinalTraversalCost(const void* state) const {
  int slen = StateSize(state);
  int len = slen + 2;
  buffer.resize(len + 1);
  buffer[len] = kNONE;
  buffer[len-1] = kSTART;
  const int* astate = reinterpret_cast<const WordID*>(state);
  int i = len - 2;
  for (int j = 0; j < slen; ++j,--i)
    buffer[i] = astate[j];
  buffer[i] = kSTOP;
  assert(i == 0);
  return ProbNoRemnant(len - 1, len);
}

SimplePair FF_LBLLM::LookupProbForBufferContents(int i) const {
  double p = WordProb(buffer[i], &buffer[i+1]);
  return SimplePair(p, buffer[i] == kUNKNOWN);
}

double FF_LBLLM::WordProb(WordIndex word, const WordIndex* history) const {
  vector<WordIndex> context;
  for (int i = 0; i < kORDER - 1 && history && (*history != kNONE); ++i) {
    context.push_back(*history++);
  }

  ProbingModel::State state, out;
  if (!context.empty() && context.back() == kSTART) {
    state = model->BeginSentenceState();
  } else {
    state = model->NullContextState();
  }

  for (int i = context.size() - 1; i >= 0; --i) {
    assert(context[i] != kSTAR);
  }

  for (int i = context.size() - 1; i >= 0; --i) {
    model->FullScore(state, context[i], out);
    state = out;
  }

  return model->FullScore(state, word, out).prob;
}

SimplePair FF_LBLLM::EstimateProb(const void* state) const {
  int len = StateSize(state);
  buffer.resize(len + 1);
  buffer[len] = kNONE;
  const int* astate = reinterpret_cast<const WordID*>(state);
  int i = len - 1;
  for (int j = 0; j < len; ++j,--i) {
    buffer[i] = astate[j];
  }

  // cout << "final state: ";
  // PrintBuffer(buffer, mapper);

  return ProbNoRemnant(len - 1, len);
}

SimplePair FF_LBLLM::ProbNoRemnant(int i, int len) const {
  int edge = len;
  bool flag = true;
  SimplePair sum;
  while (i >= 0) {
    if (buffer[i] == kSTAR) {
      edge = i;
      flag = false;
    } else if (buffer[i] == kSTART) {
      edge = i;
      flag = true;
    } else {
      if ((edge-i >= kORDER) || (flag && !(i == (len-1) && buffer[i] == kSTART)))
        sum += LookupProbForBufferContents(i);
    }
    --i;
  }
  return sum;
}

int FF_LBLLM::StateSize(const void* state) const {
  return *(static_cast<const char*>(state) + stateOffset);
}

void FF_LBLLM::SetStateSize(int size, void* state) const {
  *(static_cast<char*>(state) + stateOffset) = size;
}

int FF_LBLLM::ReserveStateSize() const {
  return (2 * (kORDER - 1) + 1) * sizeof(WordIndex) + 1;
}

/*
  FF_LBLLM(
      const string& filename, const string& feature_name,
      const bool& cache_queries)
      : fid(FD::Convert(feature_name)),
        fidOOV(FD::Convert(feature_name + "_OOV")),
        processIdentifier("FF_LBLLM"),
        cacheQueries(cache_queries), cacheHits(0), totalHits(0) {
    loadLanguageModel(filename);

    // Note: This is a hack due to lack of time.
    // Ideally, we would like a to have client server architecture, where the
    // server contains both the LM and the n-gram cache, preventing these huge
    // data structures from being replicated to every process. Also, that
    // approach would not require us to save the n-gram cache to disk after
    // every MIRA iteration.
    if (cacheQueries) {
      processId = processIdentifier.reserveId();
      cerr << "Reserved id " << processId
           << " at time " << Clock::to_time_t(GetTime()) << endl;
      cacheFile = filename + "." + to_string(processId) + ".cache.bin";
      if (boost::filesystem::exists(cacheFile)) {
        ifstream f(cacheFile);
        boost::archive::binary_iarchive ia(f);
        cerr << "Loading n-gram probability cache from " << cacheFile << endl;
        ia >> cache;
        cerr << "Finished loading " << cache.size()
             << " n-gram probabilities..." << endl;
      } else {
        cerr << "Cache file not found..." << endl;
      }
    }

    cerr << "Initializing map contents (map size=" << dict.max() << ")\n";
    for (int i = 1; i < dict.max(); ++i)
      AddToWordMap(i);
    cerr << "Done.\n";
    stateOffset = OrderToStateSize(kORDER)-1;  // offset of "state size" member
    FeatureFunction::SetStateSize(OrderToStateSize(kORDER));
    kSTART = dict.Convert("<s>");
    kSTOP = dict.Convert("</s>");
    kUNKNOWN = dict.Convert("<unk>");
    kNONE = -1;
    kSTAR = dict.Convert("<{STAR}>");
    last_id = 0;
  }

  virtual void PrepareForInput(const SentenceMetadata& smeta) {
    unsigned id = smeta.GetSentenceID();
    if (last_id > id) {
      cerr << "last_id = " << last_id << " but id = " << id << endl;
      abort();
    }
    last_id = id;
    lm->clear_cache();
  }

  inline void AddToWordMap(const WordID lbl_id) {
    const unsigned cdec_id = TD::Convert(dict.Convert(lbl_id));
    assert(cdec_id > 0);
    if (cdec_id >= cdec2lbl.size())
      cdec2lbl.resize(cdec_id + 1);
    cdec2lbl[cdec_id] = lbl_id;
  }


 private:
  void loadLanguageModel(const string& filename) {
    Time 
    WOordID kSTAR;start_time = GetTime();
    cerr << "Reading LM from " << filename << "..." << endl;
    ifstream ifile(filename);
    if (!ifile.good()) {
      cerr << "Failed to open " << filename << " for reading" << endl;
      abort();
    }
    boost::archive::binary_iarchive ia(ifile);
    ia >> lm;
    dict = lm->label_set();
    Time stop_time = GetTime();
    cerr << "Reading language model took " << GetDuration(start_time, stop_time)
         << " seconds..." << endl;
  }

  inline void SetStateSize(int size, void* state) const {
    *(static_cast<char*>(state) + stateOffset) = size;
  }


  // first = prob, second = unk
  inline SimplePair LookupProbForBufferContents(int i) const {
    if (buffer[i] == kUNKNOWN)
      return SimplePair(0.0, 1.0);
    double p = WordProb(buffer[i], &buffer[i+1]);
    return SimplePair(p, 0.0);
  }

  SimplePair EstimateProb(const vector<WordID>& phrase) const {
    cerr << "EstimateProb(&phrase): ";
    int len = phrase.size();
    buffer.resize(len + 1);
    buffer[len] = kNONE;
    int i = len - 1;
    for (int j = 0; j < len; ++j,--i)
      buffer[i] = phrase[j];
    return ProbNoRemnant(len - 1, len);
  }

  //Vocab_None is (unsigned)-1 in srilm, same as kNONE. in srilm (-1), or that SRILM otherwise interprets -1 as a terminator and not a word

  // for <s> (n-1 left words) and (n-1 right words) </s>

  static int OrderToStateSize(int order) {
    return order>1 ?
      ((order-1) * 2 + 1) * sizeof(WordID) + 1
      : 0;
  }

  virtual ~FF_LBLLM() {
    if (cacheQueries) {
      cerr << "Cache hit ratio: " << Real(cacheHits) / totalHits << endl;

      ofstream f(cacheFile);
      boost::archive::binary_oarchive ia(f);
      cerr << "Saving n-gram probability cache to " << cacheFile << endl;
      ia << cache;
      cerr << "Finished saving " << cache.size()
           << " n-gram probabilities..." << endl;

      processIdentifier.freeId(processId);
      cerr << "Freed id " << processId
           << " at time " << Clock::to_time_t(GetTime()) << endl;
    }
  }

  oxlm::Dict dict;
  mutable vector<WordID> buffer;
  int stateOffset;
  WordID kSTART;
  WordID kSTOP;
  WordID kUNKNOWN;
  WordID kNONE;
  WordID kSTAR;
  const int fid;
  const int fidOOV;
  vector<int> cdec2lbl;
  unsigned last_id;

  boost::shared_ptr<FactoredNLM> lm;

  ProcessIdentifier processIdentifier;
  int processId;

  bool cacheQueries;
  string cacheFile;
  mutable QueryCache cache;
  mutable int cacheHits, totalHits;
};

*/

boost::shared_ptr<FeatureFunction> LBLLanguageModelFactory::Create(string param) const {
  string filename, featname;
  ParseOptions(param, filename, featname);
  return boost::make_shared<FF_LBLLM>(filename, featname);
}

string LBLLanguageModelFactory::usage(bool params, bool verbose) const {
  return "";
}

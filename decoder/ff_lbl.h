#pragma once

#include <vector>

#include "ff_factory.h"
#include "ff.h"
#include "hg.h"
#include "trule.h"
#include "lm/enumerate_vocab.hh"
#include "lm/model.hh"

class CdecMapper : public lm::EnumerateVocab {
 public:
  virtual void Add(lm::WordIndex word_id, const StringPiece& word);

  lm::WordIndex MapWord(WordID cdec_word_id) const;

  size_t size() const;

  StringPiece getWord(lm::WordIndex word_id) const;

 private:
  std::vector<lm::WordIndex> cdec2lbl;
  std::vector<std::string> words;
};

struct SimplePair {
  SimplePair();

  SimplePair(double x, double y);

  SimplePair& operator+=(const SimplePair& o);

  double first;
  double second;
};

class FF_LBLLM : public FeatureFunction {
 public:
  FF_LBLLM(const std::string& filename, const std::string& featname);

 protected:
  virtual void TraversalFeaturesImpl(
    const SentenceMetadata&, const HG::Edge& edge,
    const std::vector<const void*>& ant_states, SparseVector<double>* features,
    SparseVector<double>* estimated_features, void* state) const;

  virtual void FinalTraversalFeatures(
      const void* ant_state, SparseVector<double>* features) const;

 private:
  SimplePair LookupWords(
      const TRule& rule, const std::vector<const void*>& ant_states,
      void* vstate) const;

  SimplePair FinalTraversalCost(const void* state) const;

  SimplePair EstimateProb(void* state) const;

  int StateSize(const void* state) const;

  void SetStateSize(int size, void* state) const;

  int ReserveStateSize() const;

  SimplePair LookupProbForBufferContents(int i) const;

  SimplePair ProbNoRemnant(int i, int len) const;

  double WordProb(lm::WordIndex word, const lm::WordIndex* history) const;

  CdecMapper mapper;
  boost::shared_ptr<lm::ngram::ProbingModel> model;
  int fid, fidOOV;
  int stateOffset;
  mutable std::vector<lm::WordIndex> buffer;
  mutable bool debug;

  lm::WordIndex kNONE;
  lm::WordIndex kUNKNOWN;
  lm::WordIndex kSTAR;
  lm::WordIndex kSTART;
  lm::WordIndex kSTOP;
};

struct LBLLanguageModelFactory : public FactoryBase<FeatureFunction> {
  FP Create(std::string param) const;

  std::string usage(bool params, bool verbose) const;
};

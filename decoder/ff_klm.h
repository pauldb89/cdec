#ifndef _KLM_FF_H_
#define _KLM_FF_H_

#include <vector>
#include <string>

#include "ff_factory.h"
#include "ff.h"
#include "lm/enumerate_vocab.hh"

template <class Model> struct KLanguageModelImpl;

struct VMapper : public lm::EnumerateVocab {
  VMapper(std::vector<lm::WordIndex>* out);

  void Add(lm::WordIndex index, const StringPiece &str);

  StringPiece getWord(lm::WordIndex word_id) const;

  std::vector<std::string> getWords() const;

  std::vector<std::string> words;
  std::vector<lm::WordIndex>* out_;
  const lm::WordIndex kLM_UNKNOWN_TOKEN;
};

// the supported template types are instantiated explicitly
// in ff_klm.cc.
template <class Model>
class KLanguageModel : public FeatureFunction {
 public:
  // param = "filename.lm [-o n]"
  KLanguageModel(const std::string& param);
  ~KLanguageModel();
  virtual void FinalTraversalFeatures(const void* context,
                                      SparseVector<double>* features) const;
  static std::string usage(bool param,bool verbose);
 protected:
  virtual void TraversalFeaturesImpl(const SentenceMetadata& smeta,
                                     const HG::Edge& edge,
                                     const std::vector<const void*>& ant_contexts,
                                     SparseVector<double>* features,
                                     SparseVector<double>* estimated_features,
                                     void* out_context) const;
 private:
  int fid_;        // LanguageModel
  int oov_fid_;    // LanguageModel_OOV
  int emit_fid_;   // LanguageModel_Emit [only used for class-based LMs]
  KLanguageModelImpl<Model>* pimpl_;
};

struct KLanguageModelFactory : public FactoryBase<FeatureFunction> {
  FP Create(std::string param) const;
  std::string usage(bool params,bool verbose) const;
};

#endif

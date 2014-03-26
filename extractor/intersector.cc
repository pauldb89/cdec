#include "intersector.h"

#include "data_array.h"
#include "matching_comparator.h"
#include "phrase.h"
#include "phrase_location.h"
#include "precomputation.h"
#include "suffix_array.h"
#include "time_util.h"
#include "veb.h"
#include "vocabulary.h"

namespace extractor {

Intersector::Intersector(shared_ptr<Vocabulary> vocabulary,
                         shared_ptr<Precomputation> precomputation,
                         shared_ptr<SuffixArray> suffix_array,
                         shared_ptr<MatchingComparator> comparator,
                         bool use_baeza_yates) :
    vocabulary(vocabulary),
    precomputation(precomputation),
    suffix_array(suffix_array),
    use_baeza_yates(use_baeza_yates) {
  shared_ptr<DataArray> data_array = suffix_array->GetData();
  linear_merger = make_shared<LinearMerger>(vocabulary, data_array, comparator);
  binary_search_merger = make_shared<BinarySearchMerger>(
      vocabulary, linear_merger, data_array, comparator);
}

Intersector::Intersector(shared_ptr<Vocabulary> vocabulary,
                         shared_ptr<Precomputation> precomputation,
                         shared_ptr<SuffixArray> suffix_array,
                         shared_ptr<LinearMerger> linear_merger,
                         shared_ptr<BinarySearchMerger> binary_search_merger,
                         bool use_baeza_yates) :
    vocabulary(vocabulary),
    precomputation(precomputation),
    suffix_array(suffix_array),
    linear_merger(linear_merger),
    binary_search_merger(binary_search_merger),
    use_baeza_yates(use_baeza_yates) {}

Intersector::Intersector() {}

Intersector::~Intersector() {}

PhraseLocation Intersector::Intersect(
    const Phrase& prefix, PhraseLocation& prefix_location,
    const Phrase& suffix, PhraseLocation& suffix_location,
    const Phrase& phrase) {
  vector<int> symbols = phrase.Get();

  // We should never attempt to do an intersect query for a pattern starting or
  // ending with a non terminal. The RuleFactory should handle these cases,
  // initializing the matchings list with the one for the pattern without the
  // starting or ending terminal.
  assert(vocabulary->IsTerminal(symbols.front())
      && vocabulary->IsTerminal(symbols.back()));

  if (precomputation->ContainsCollocation(symbols)) {
    return PhraseLocation(
        precomputation->GetCollocationMatches(symbols), phrase.Arity() + 1);
  }

  vector<int> locations;
  auto start_time = Clock::now();
  ExtendPhraseLocation(prefix, prefix_location);
  ExtendPhraseLocation(suffix, suffix_location);
  sort_time += GetDuration(start_time, Clock::now());

  auto function_start = Clock::now();
  shared_ptr<vector<int> > prefix_matchings = prefix_location.matchings;
  shared_ptr<vector<int> > suffix_matchings = suffix_location.matchings;
  int prefix_subpatterns = prefix_location.num_subpatterns;
  int suffix_subpatterns = suffix_location.num_subpatterns;
  if (use_baeza_yates) {
    auto start_time = Clock::now();
    binary_search_merger->Merge(locations, phrase, suffix,
        prefix_matchings->begin(), prefix_matchings->end(),
        suffix_matchings->begin(), suffix_matchings->end(),
        prefix_subpatterns, suffix_subpatterns);
    auto stop_time = Clock::now();
    binary_search_merger->binary_search_time +=
        GetDuration(start_time, stop_time);
  } else {
    linear_merger->Merge(locations, phrase, suffix, prefix_matchings->begin(),
        prefix_matchings->end(), suffix_matchings->begin(),
        suffix_matchings->end(), prefix_subpatterns, suffix_subpatterns);
  }

  auto function_stop = Clock::now();
  inner_time += GetDuration(function_start, function_stop);
  return PhraseLocation(locations, phrase.Arity() + 1);
}

void Intersector::ExtendPhraseLocation(
    const Phrase& phrase, PhraseLocation& phrase_location) {
  int low = phrase_location.sa_low, high = phrase_location.sa_high;
  if (phrase_location.matchings != NULL) {
    return;
  }

  phrase_location.num_subpatterns = 1;
  phrase_location.sa_low = phrase_location.sa_high = 0;

  vector<int> symbols = phrase.Get();
  if (precomputation->ContainsContiguousPhrase(symbols)) {
    phrase_location.matchings = make_shared<vector<int>>(
        precomputation->GetContiguousMatches(symbols));
    return;
  }

  vector<int> matchings;
  matchings.reserve(high - low + 1);
  shared_ptr<VEB> veb = VEB::Create(suffix_array->GetSize());
  for (int i = low; i < high; ++i) {
    veb->Insert(suffix_array->GetSuffix(i));
  }

  int value = veb->GetMinimum();
  while (value != -1) {
    matchings.push_back(value);
    value = veb->GetSuccessor(value);
  }

  phrase_location.matchings = make_shared<vector<int> >(matchings);
}

} // namespace extractor

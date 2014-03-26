#include "linear_merger.h"

#include <cmath>

#include "data_array.h"
#include "matching.h"
#include "matching_comparator.h"
#include "phrase.h"
#include "phrase_location.h"
#include "time_util.h"
#include "vocabulary.h"

namespace extractor {

LinearMerger::LinearMerger(shared_ptr<Vocabulary> vocabulary,
                           shared_ptr<DataArray> data_array,
                           shared_ptr<MatchingComparator> comparator) :
    vocabulary(vocabulary), data_array(data_array), comparator(comparator),
    linear_merge_time(0) {}

LinearMerger::LinearMerger() {}

LinearMerger::~LinearMerger() {}

void LinearMerger::Merge(
    vector<int>& locations, const Phrase& phrase, const Phrase& suffix,
    vector<int>::iterator prefix_start, vector<int>::iterator prefix_end,
    vector<int>::iterator suffix_start, vector<int>::iterator suffix_end,
    int prefix_subpatterns, int suffix_subpatterns) {
  auto start_time = Clock::now();
  int last_chunk_len = suffix.GetChunkLen(suffix.Arity());
  bool offset = !vocabulary->IsTerminal(suffix.GetSymbol(0));

  while (prefix_start != prefix_end) {
    Matching left(prefix_start, prefix_subpatterns,
        data_array->GetSentenceId(*prefix_start));

    while (suffix_start != suffix_end) {
      Matching right(suffix_start, suffix_subpatterns,
          data_array->GetSentenceId(*suffix_start));
      if (comparator->Compare(left, right, last_chunk_len, offset) > 0) {
        suffix_start += suffix_subpatterns;
      } else {
        break;
      }
    }

    int start_position = *prefix_start;
    vector<int> :: iterator i = suffix_start;
    while (prefix_start != prefix_end && *prefix_start == start_position) {
      Matching left(prefix_start, prefix_subpatterns,
          data_array->GetSentenceId(*prefix_start));

      while (i != suffix_end) {
        Matching right(i, suffix_subpatterns, data_array->GetSentenceId(*i));
        int comparison = comparator->Compare(left, right, last_chunk_len,
                                             offset);
        if (comparison == 0) {
          vector<int> merged = left.Merge(right, phrase.Arity() + 1);
          locations.insert(locations.end(), merged.begin(), merged.end());
        } else if (comparison < 0) {
          break;
        }
        i += suffix_subpatterns;
      }

      prefix_start += prefix_subpatterns;
    }
  }
  auto stop_time = Clock::now();
  linear_merge_time += GetDuration(start_time, stop_time);
}

} // namespace extractor

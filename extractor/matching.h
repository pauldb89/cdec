#ifndef _MATCHING_H_
#define _MATCHING_H_

#include <memory>
#include <vector>

using namespace std;

namespace extractor {

struct Matching {
  Matching(vector<int>::iterator start, int len, int sentence_id);

  vector<int> Merge(const Matching& other, int num_subpatterns) const;

  vector<int> positions;
  int sentence_id;
};

} // namespace extractor

#endif

#include <gmock/gmock.h>

#include "precomputation.h"

namespace extractor {

class MockPrecomputation : public Precomputation {
 public:
  MOCK_CONST_METHOD1(Contains, bool(const vector<int>& pattern));
  MOCK_CONST_METHOD1(GetCollocations, const vector<int>&(const vector<int>& pattern));
  MOCK_CONST_METHOD1(ContainsContiguousPhrase, bool(const vector<int>& pattern));
  MOCK_CONST_METHOD1(GetContiguousMatches, const vector<int>&(const vector<int>& pattern));
  MOCK_CONST_METHOD1(ContainsCollocation, bool(const vector<int>& pattern));
  MOCK_CONST_METHOD1(GetCollocationMatches, const vector<int>&(const vector<int>& pattern));
};

} // namespace extractor

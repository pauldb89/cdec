#include <gmock/gmock.h>

#include "precomputation.h"

namespace extractor {

class MockPrecomputation : public Precomputation {
 public:
  MOCK_CONST_METHOD1(Contains, bool(const vector<int>&));
  MOCK_CONST_METHOD1(GetCollocations, shared_ptr<vector<int>>(const vector<int>&));
};

} // namespace extractor

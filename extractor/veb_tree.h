#ifndef _VEB_TREE_H_
#define _VEB_TREE_H_

#include <memory>
#include <vector>

#include "veb.h"

using namespace std;

namespace extractor {

class VEBTree: public VEB {
 public:
  VEBTree(int size);

  void Insert(int value);

  int GetSuccessor(int value);

 private:
  int GetNextValue(int value);
  int GetCluster(int value);
  int Compose(int cluster, int value);

  int lower_bits, upper_size;
  shared_ptr<VEB> summary;
  vector<shared_ptr<VEB> > clusters;
};

} // namespace extractor

#endif

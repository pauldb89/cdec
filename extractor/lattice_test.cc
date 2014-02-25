#include <gtest/gtest.h>

#include <memory>

#include "lattice.h"
#include "mocks/mock_vocabulary.h"

using namespace std;
using namespace ::testing;

namespace extractor {
namespace {

class LatticeTest : public Test {
 protected:
  virtual void SetUp() {
    vocabulary = make_shared<MockVocabulary>();
    for (int i = 0; i < 10; ++i) {
      string word(1, i + 'a');
      EXPECT_CALL(*vocabulary, GetTerminalValue(i + 2))
          .WillRepeatedly(Return(word));
      EXPECT_CALL(*vocabulary, GetTerminalIndex(word))
          .WillRepeatedly(Return(i + 2));
    }

    EXPECT_CALL(*vocabulary, GetTerminalValue(100))
        .WillRepeatedly(Return("*EPS*"));
    EXPECT_CALL(*vocabulary, GetTerminalIndex("*EPS*"))
        .WillRepeatedly(Return(100));
    EXPECT_CALL(*vocabulary, GetTerminalValue(200))
        .WillRepeatedly(Return("n\'t"));
    EXPECT_CALL(*vocabulary, GetTerminalIndex("n\'t"))
        .WillRepeatedly(Return(200));
    EXPECT_CALL(*vocabulary, GetTerminalValue(201))
        .WillRepeatedly(Return("a\\b"));
    EXPECT_CALL(*vocabulary, GetTerminalIndex("a\\b"))
        .WillRepeatedly(Return(201));
  }

  shared_ptr<MockVocabulary> vocabulary;
};

TEST_F(LatticeTest, TestSentence) {
  string input = "a b c";
  Lattice lattice(input, vocabulary, true);

  EXPECT_EQ(4, lattice.GetSize());

  vector<pair<int, int>> expected_transitions;
  for (int i = 0; i < 3; ++i) {
    expected_transitions = {{i + 1, i + 2}};
    EXPECT_EQ(expected_transitions, lattice.GetTransitions(i));
  }
  expected_transitions = {};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(3));

  vector<pair<int, int>> expected_extensions = {{1, 1}, {2, 2}, {3, 3}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(0, 1, 10));
  expected_extensions = {{1, 1}, {2, 2}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(0, 1, 2));
  expected_extensions = {{2, 2}, {3, 3}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(0, 2, 3));
  expected_extensions = {{2, 1}, {3, 2}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(1, 1, 3));
  expected_extensions = {{2, 1}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(1, 1, 1));
  expected_extensions = {{3, 1}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(2, 1, 5));
  expected_extensions = {};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(2, 2, 3));
}

TEST_F(LatticeTest, TestDAG) {
  string input = "((('a', 0, 1),),(('b', 0, 2),('c', -0.2, 1),('d', -1.5, 1),),"
                 "(('e', 0, 1),),(('f', 0, 2),('g', -2.0, 1),),"
                 "(('h', -1.0, 1),),)";
  Lattice lattice(input, vocabulary, false);

  EXPECT_EQ(6, lattice.GetSize());

  vector<pair<int, int>> expected_transitions = {{1, 2}};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(0));
  expected_transitions = {{3, 3}, {2, 4}, {2, 5}};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(1));
  expected_transitions = {{3, 6}};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(2));
  expected_transitions = {{5, 7}, {4, 8}};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(3));
  expected_transitions = {{5, 9}};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(4));
  expected_transitions = {};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(5));

  vector<pair<int, int>> expected_extensions =
      {{1, 1}, {2, 2}, {3, 2}, {4, 3}, {5, 3}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(0, 1, 10));
  expected_extensions = {{1, 1}, {2, 2}, {3, 2}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(0, 1, 2));
  expected_extensions = {{4, 3}, {5, 3}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(0, 3, 5));
  expected_extensions = {{2, 1}, {3, 1}, {4, 2}, {5, 2}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(1, 1, 2));
  expected_extensions = {{2, 1}, {3, 1}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(1, 1, 1));
  expected_extensions = {{4, 2}, {5, 2}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(1, 2, 2));
  expected_extensions = {{3, 1}, {4, 2}, {5, 2}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(2, 1, 2));
  expected_extensions = {{4, 1}, {5, 1}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(3, 1, 1));
  expected_extensions = {{5, 1}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(4, 1, 1));
  expected_extensions = {};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(5, 1, 10));
}

TEST_F(LatticeTest, TestEpsilonTransitions) {
  string input = "((('a', 0, 1),('*EPS*', 0, 2),),(('*EPS*', 0, 3),('b', 0, 2),),"
                 "(('c', 0, 1),),(('d', 0, 1),('e', 0, 2),),(('f', 0, 2),),"
                 "(('g', 0, 1),),)";
  Lattice lattice(input, vocabulary, false);

  EXPECT_EQ(7, lattice.GetSize());

  vector<pair<int, int>> expected_transitions = {{4, 2}, {3, 4}, {1, 2}};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(0));
  expected_transitions = {{6, 7}, {3, 3}};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(1));
  expected_transitions = {{3, 4}};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(2));
  expected_transitions = {{5, 6}, {4, 5}};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(3));
  expected_transitions = {{6, 7}};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(4));
  expected_transitions = {{6, 8}};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(5));
  expected_transitions = {};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(6));

  vector<pair<int, int>> expected_extensions =
      {{1, 1}, {3, 1}, {4, 1}, {5, 2}, {6, 2}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(0, 1, 10));
  expected_extensions = {{1, 1}, {3, 1}, {4, 1}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(0, 1, 1));
  expected_extensions = {{5, 2}, {6, 2}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(0, 2, 2));
  expected_extensions = {{3, 1}, {4, 2}, {5, 2}, {6, 1}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(1, 1, 2));
  expected_extensions = {{3, 1}, {6, 1}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(1, 1, 1));
  expected_extensions = {{4, 2}, {5, 2}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(1, 2, 2));
  expected_extensions = {{3, 1}, {4, 2}, {5, 2}, {6, 3}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(2, 1, 3));
  expected_extensions = {{4, 1}, {5, 1}, {6, 2}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(3, 1, 3));
  expected_extensions = {{6, 1}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(4, 1, 3));
  expected_extensions = {{6, 1}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(5, 1, 3));
  expected_extensions = {};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(6, 1, 3));
}

TEST_F(LatticeTest, TestChainedEpsilonTransitions) {
  string input = "((('*EPS*', 0, 1),),(('*EPS*', 0, 1),),(('a', 0, 1),),"
                 "(('*EPS*', 0, 1),),(('*EPS*', 0, 1),),)";
  Lattice lattice(input, vocabulary, false);

  EXPECT_EQ(6, lattice.GetSize());

  vector<pair<int, int>> expected_transitions = {{5, 2}, {4, 2}, {3, 2}};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(0));
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(1));
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(2));
  expected_transitions = {};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(3));
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(4));
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(5));

  vector<pair<int, int>> expected_extensions = {{3, 1}, {4, 1}, {5, 1}};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(0, 1, 10));
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(1, 1, 10));
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(2, 1, 10));
  expected_extensions = {};
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(3, 1, 10));
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(4, 1, 10));
  EXPECT_EQ(expected_extensions, lattice.GetExtensions(5, 1, 10));
}

TEST_F(LatticeTest, TestEscapedSymbols) {
  string input = "((('n\\\'t', 0, 1),),(('a\\\\b', 0, 1),),)";
  Lattice lattice(input, vocabulary, false);

  EXPECT_EQ(3, lattice.GetSize());
  vector<pair<int, int>> expected_transitions = {{1, 200}};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(0));
  expected_transitions = {{2, 201}};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(1));
  expected_transitions = {};
  EXPECT_EQ(expected_transitions, lattice.GetTransitions(2));
}

} // namespace
} // namespace extractor

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <vector>

#include "grammar.h"
#include "grammar_extractor.h"
#include "lattice.h"
#include "mocks/mock_fast_intersector.h"
#include "mocks/mock_matchings_finder.h"
#include "mocks/mock_rule_extractor.h"
#include "mocks/mock_sampler.h"
#include "mocks/mock_scorer.h"
#include "mocks/mock_vocabulary.h"
#include "phrase_builder.h"
#include "phrase_location.h"

using namespace std;
using namespace ::testing;

namespace extractor {
namespace {

class GrammarExtractorTest : public Test {
 protected:
  virtual void SetUp() {
    finder = make_shared<MockMatchingsFinder>();
    fast_intersector = make_shared<MockFastIntersector>();

    vocabulary = make_shared<MockVocabulary>();
    EXPECT_CALL(*vocabulary, GetTerminalValue(2)).WillRepeatedly(Return("a"));
    EXPECT_CALL(*vocabulary, GetTerminalValue(3)).WillRepeatedly(Return("b"));
    EXPECT_CALL(*vocabulary, GetTerminalValue(4)).WillRepeatedly(Return("c"));
    EXPECT_CALL(*vocabulary, GetTerminalValue(100))
        .WillRepeatedly(Return("*EPS*"));
    EXPECT_CALL(*vocabulary, GetTerminalIndex("a")).WillRepeatedly(Return(2));
    EXPECT_CALL(*vocabulary, GetTerminalIndex("b")).WillRepeatedly(Return(3));
    EXPECT_CALL(*vocabulary, GetTerminalIndex("c")).WillRepeatedly(Return(4));
    EXPECT_CALL(*vocabulary, GetTerminalIndex("*EPS*"))
        .WillRepeatedly(Return(100));

    phrase_builder = make_shared<PhraseBuilder>(vocabulary);

    scorer = make_shared<MockScorer>();
    feature_names = {"f1"};
    EXPECT_CALL(*scorer, GetFeatureNames())
        .WillRepeatedly(Return(feature_names));

    sampler = make_shared<MockSampler>();
    EXPECT_CALL(*sampler, Sample(_, _))
        .WillRepeatedly(Return(PhraseLocation(0, 1)));

    Phrase phrase;
    vector<double> scores = {0.5};
    vector<pair<int, int>> phrase_alignment = {make_pair(0, 0)};
    vector<Rule> rules = {Rule(phrase, phrase, scores, phrase_alignment)};
    rule_extractor = make_shared<MockRuleExtractor>();
    EXPECT_CALL(*rule_extractor, ExtractRules(_, _))
        .WillRepeatedly(Return(rules));

    grammar_extractor = make_shared<GrammarExtractor>(
        finder, fast_intersector, phrase_builder, rule_extractor, vocabulary,
        sampler, scorer, 1, 10, 2, 3, 5);
  }

  void CheckInput(
      const string& input, bool sentence,
      int contiguous_rules, int discontiguous_rules) {
    EXPECT_CALL(*finder, Find(_, _, _))
        .Times(contiguous_rules)
        .WillRepeatedly(Return(PhraseLocation(0, 1)));
    EXPECT_CALL(*fast_intersector, Intersect(_, _, _))
        .Times(discontiguous_rules)
        .WillRepeatedly(Return(PhraseLocation(0, 1)));

    Lattice lattice(input, vocabulary, sentence);
    unordered_set<int> blacklisted_sentence_ids;
    Grammar grammar = grammar_extractor->GetGrammar(
        lattice, blacklisted_sentence_ids);

    EXPECT_EQ(feature_names, grammar.GetFeatureNames());
    EXPECT_EQ(contiguous_rules + discontiguous_rules,
              grammar.GetRules().size());
  }

  vector<string> feature_names;
  shared_ptr<MockMatchingsFinder> finder;
  shared_ptr<MockFastIntersector> fast_intersector;
  shared_ptr<MockVocabulary> vocabulary;
  shared_ptr<PhraseBuilder> phrase_builder;
  shared_ptr<MockScorer> scorer;
  shared_ptr<MockSampler> sampler;
  shared_ptr<MockRuleExtractor> rule_extractor;
  shared_ptr<GrammarExtractor> grammar_extractor;
};

TEST_F(GrammarExtractorTest, TestGetGrammarDifferentWords) {
  CheckInput("a b c", true, 6, 1);
}

TEST_F(GrammarExtractorTest, TestGetGrammarRepeatingWords) {
  CheckInput("a b c a b", true, 12, 16);
}

TEST_F(GrammarExtractorTest, TestLattice) {
  string input = "((('a', 0, 1),('b', 0, 2),),(('c', 0, 3),('b', 0, 2),),"
                 "(('a', 0, 1),),(('a', 0, 1),('a', 0, 2),),(('a', 0, 2),),"
                 "(('c', 0, 1),),)";
  CheckInput(input, false, 18, 12);
}

TEST_F(GrammarExtractorTest, TestEpsilonTransitions) {
  string input = "((('a', 0, 1),('*EPS*', 0, 2),),(('*EPS*', 0, 3),('b', 0, 2),),"
                 "(('a', 0, 1),),(('b', 0, 1),('c', 0, 2),),(('c', 0, 2),),"
                 "(('a', 0, 1),),)";
  CheckInput(input, false, 15, 9);
}

} // namespace
} // namespace extractor

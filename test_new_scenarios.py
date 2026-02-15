"""Test that all demo scenario phrases trigger detection appropriately.

Tests coverage of Tier 1 phrases and Tier 2 scenarios across all demo scripts.
Each scam scenario should trigger detection on the FIRST or SECOND suspicious line.
"""

import sys
sys.path.insert(0, '.')
from src.core.content_analyzer import ContentAnalyzer

analyzer = ContentAnalyzer()

# Format: (phrase, expected_level, scenario_name)
# expected_level: "low", "medium", "high"
# For "medium": accept "medium" or "high"
# For "high": only accept "high"
# For "low": only accept "low"

TEST_CASES = [
    # =========================================================================
    # GIFT CARD SCAM - Dashboard scenario lines
    # =========================================================================
    ("Yes, I can go to Walgreens right now.", "medium", "Gift Card Line 1"),
    ("How many gift cards do you need?", "medium", "Gift Card Line 2"),
    ("Okay, five hundred dollars in Google Play cards.", "medium", "Gift Card Line 3"),
    ("Should I scratch off the back and read you the numbers?", "high", "Gift Card Line 4"),
    
    # =========================================================================
    # IRS IMPERSONATION SCAM
    # Note: "The IRS?" triggers semantic match - this is intentional, IRS mention is suspicious
    # =========================================================================
    ("The IRS? Oh my, what did I do wrong?", "medium", "IRS Line 1 (IRS mention)"),
    ("You need my social security number to verify?", "medium", "IRS Line 2"),
    ("Okay, my social security number is 4-8-3...", "high", "IRS Line 3"),
    
    # =========================================================================
    # TECH SUPPORT SCAM
    # Note: "virus" triggers semantic match - this is appropriate for scam detection
    # =========================================================================
    ("My computer has a virus?", "medium", "Tech Support Line 1 (virus mention)"),
    ("TeamViewer? Let me download that for you.", "high", "Tech Support Line 2"),
    ("Okay, I'm giving you remote access now.", "high", "Tech Support Line 3"),
    ("It's asking for my password now.", "low", "Tech Support Line 4 (generic)"),
    ("Three hundred dollars to fix it? Let me get my credit card.", "medium", "Tech Support Line 5"),
    
    # =========================================================================
    # GRANDCHILD EMERGENCY SCAM
    # Note: "jail" triggers semantic match - appropriate for scam detection
    # =========================================================================
    ("Tommy? Is that you? You sound different.", "low", "Grandchild Line 1"),
    ("Oh my god, are you hurt? What happened?", "low", "Grandchild Line 2"),
    ("You're in jail? How did this happen?", "medium", "Grandchild Line 3 (jail mention)"),
    ("Of course I'll help you with bail money, sweetie.", "medium", "Grandchild Line 4"),
    ("Don't worry, I won't tell your parents about this.", "high", "Grandchild Line 5"),
    ("How much do you need for the bail?", "medium", "Grandchild Line 6"),
    ("I can go to the store and get that for you.", "low", "Grandchild Line 7"),
    
    # =========================================================================
    # ROMANCE SCAM
    # Note: "do anything for you" is a vulnerability indicator, appropriate to flag
    # =========================================================================
    ("I feel so lucky we found each other.", "low", "Romance Line 1"),
    ("Of course I trust you, darling.", "low", "Romance Line 2"),
    ("You're stuck overseas? That's terrible.", "medium", "Romance Line 3"),
    ("How much do you need to get home?", "medium", "Romance Line 4"),
    ("Western Union? I can figure out how to do that.", "high", "Romance Line 5"),
    ("I'd do anything for you, you know that.", "high", "Romance Line 6 (vulnerability)"),
    
    # =========================================================================
    # SSN SUSPENSION SCAM
    # =========================================================================
    ("My social security number is suspended?", "high", "SSN Line 1"),
    ("I don't want to be arrested, please help me.", "medium", "SSN Line 2"),
    ("I'll go get the gift cards right now to pay the fine.", "high", "SSN Line 3"),
    
    # =========================================================================
    # UTILITY SHUTOFF SCAM
    # =========================================================================
    ("You're going to shut off my electricity today?", "medium", "Utility Line 1"),
    ("I have to pay right now?", "medium", "Utility Line 2"),
    ("There's a Bitcoin ATM at the gas station I can use?", "high", "Utility Line 3"),
    
    # =========================================================================
    # BANK FRAUD ALERT SCAM
    # =========================================================================
    ("Someone's using my account fraudulently?", "medium", "Bank Fraud Line 1"),
    ("Yes, I'll verify my information with you.", "low", "Bank Fraud Line 2"),
    ("I need to wire money to a safe account you control?", "high", "Bank Fraud Line 3"),
    ("A verification code just came to my phone.", "medium", "Bank Fraud Line 4"),
    ("You want me to read it to you? Okay, it says...", "high", "Bank Fraud Line 5"),
    
    # =========================================================================
    # LOTTERY WINNER SCAM
    # =========================================================================
    ("I won a million dollars? Really?", "medium", "Lottery Line 1"),
    ("A processing fee? How much?", "high", "Lottery Line 2"),
    ("Two thousand dollars in gift cards to claim my prize?", "high", "Lottery Line 3"),
    
    # =========================================================================
    # MEDICARE SCAM
    # Note: Medicare mentions trigger detection - appropriate since unsolicited Medicare calls are common scams
    # =========================================================================
    ("Yes, I'm on Medicare.", "medium", "Medicare Line 1 (Medicare context)"),
    ("A new card? I didn't know I needed one.", "medium", "Medicare Line 2 (new card)"),
    ("My Medicare number? Let me get my card.", "medium", "Medicare Line 3"),
    ("And you need my bank account for the deposit?", "high", "Medicare Line 4"),
    
    # =========================================================================
    # BENIGN SCENARIOS - Should NOT trigger (must stay LOW)
    # =========================================================================
    ("I'm at Target buying a gift card for my grandson's birthday.", "low", "Benign: Birthday Gift"),
    ("He loves video games so I'm getting him a PlayStation card.", "low", "Benign: Video Game Gift"),
    ("Fifty dollars should be a nice gift, don't you think?", "low", "Benign: Gift Amount"),
    
    ("Hi sweetheart, how are the kids doing?", "low", "Benign: Family Call 1"),
    ("That's wonderful! I'm so glad you called.", "low", "Benign: Family Call 2"),
    ("Let's have dinner together this Sunday.", "low", "Benign: Family Call 3"),
    
    ("Yes, I have an appointment with Dr. Johnson at 2pm.", "low", "Benign: Doctor 1"),
    ("My prescription is ready at the pharmacy?", "low", "Benign: Doctor 2"),
    ("I'll pick it up on my way home from the checkup.", "low", "Benign: Doctor 3"),
    
    ("Yes, I called the bank myself about my account.", "low", "Benign: Legit Bank 1"),
    ("I'm looking at my statement right now.", "low", "Benign: Legit Bank 2"),
    ("The charge on February 3rd was mine, that's correct.", "low", "Benign: Legit Bank 3"),
    
    ("I'd like to order a large pepperoni pizza please.", "low", "Benign: Food Order 1"),
    ("Yes, delivery to 123 Oak Street.", "low", "Benign: Food Order 2"),
    
    ("The church potluck is this Saturday at noon.", "low", "Benign: Church 1"),
    ("I'm bringing my famous apple pie.", "low", "Benign: Church 2"),
    
    ("The plumber is coming tomorrow to fix the sink.", "low", "Benign: Home Repair"),
    ("I'll write him a check when he's done.", "low", "Benign: Payment"),
    
    ("Good morning, Helen! Beautiful weather today.", "low", "Benign: Neighbor 1"),
    ("My tomatoes are finally coming in.", "low", "Benign: Neighbor 2"),
]

def run_tests():
    """Run all test cases and report results."""
    print("=" * 70)
    print("SCAM DETECTION TEST - EXPANDED SCENARIOS")
    print("=" * 70)
    print()
    
    passed = 0
    failed = 0
    failed_tests = []
    
    for phrase, expected, scenario in TEST_CASES:
        result = analyzer.analyze(phrase)
        actual = result["risk_level"]
        
        # Determine success based on expected level
        if expected == "low":
            success = (actual == "low")
        elif expected == "medium":
            success = (actual in ["medium", "high"])
        else:  # expected == "high"
            success = (actual == "high")
        
        if success:
            passed += 1
        else:
            failed += 1
            failed_tests.append({
                "scenario": scenario,
                "phrase": phrase,
                "expected": expected,
                "actual": actual,
                "score": result["risk_score"],
                "factors": result.get("risk_factors", [])[:2],
                "tier1": result.get("detection_trigger", {}).get("match_type", ""),
            })
    
    # Print failures
    if failed_tests:
        print("FAILED TESTS:")
        print("-" * 70)
        for test in failed_tests:
            print(f"\n❌ {test['scenario']}")
            phrase_display = test['phrase'][:55] + "..." if len(test['phrase']) > 55 else test['phrase']
            print(f"   Phrase: \"{phrase_display}\"")
            print(f"   Expected: {test['expected']}, Got: {test['actual']} (score={test['score']:.2f})")
            if test['tier1']:
                print(f"   Trigger: {test['tier1']}")
            if test['factors']:
                for rf in test['factors']:
                    print(f"   → {rf[:60]}...")
        print()
    
    # Print summary
    print("=" * 70)
    total = len(TEST_CASES)
    pct = 100 * passed / total
    
    if pct >= 90:
        status = "✅ PASS"
    else:
        status = "❌ FAIL"
    
    print(f"Results: {passed}/{total} passed ({pct:.0f}%) {status}")
    
    if failed > 0:
        print(f"FAILED: {failed} tests need attention")
        
        # Categorize failures
        benign_fails = [t for t in failed_tests if "Benign" in t["scenario"]]
        scam_fails = [t for t in failed_tests if "Benign" not in t["scenario"]]
        
        if benign_fails:
            print(f"  - {len(benign_fails)} benign scenarios incorrectly flagged")
        if scam_fails:
            print(f"  - {len(scam_fails)} scam scenarios not detected")
    
    print("=" * 70)
    
    # Return success status
    return pct >= 90


def test_specific_phrase(phrase: str):
    """Test a specific phrase and print detailed results."""
    print(f"\nTesting: \"{phrase}\"")
    print("-" * 50)
    result = analyzer.analyze(phrase)
    print(f"Risk Level: {result['risk_level']}")
    print(f"Risk Score: {result['risk_score']:.2f}")
    print(f"Confidence: {result['confidence']:.2f}")
    
    if result.get("detection_trigger"):
        trigger = result["detection_trigger"]
        print(f"Trigger: {trigger.get('phrase', 'N/A')}")
        print(f"Match Type: {trigger.get('match_type', 'N/A')}")
    
    if result.get("risk_factors"):
        print("Risk Factors:")
        for rf in result["risk_factors"]:
            print(f"  - {rf}")
    
    print()
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test scam detection coverage")
    parser.add_argument("--phrase", type=str, help="Test a specific phrase")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show all test results")
    args = parser.parse_args()
    
    if args.phrase:
        test_specific_phrase(args.phrase)
    else:
        success = run_tests()
        sys.exit(0 if success else 1)

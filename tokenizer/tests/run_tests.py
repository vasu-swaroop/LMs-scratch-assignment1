"""
Comprehensive test suite for tokenizer
Run all tests: python -m tokenizer.tests.run_tests
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tokenizer.tests.test_inference import test_common_words_not_split, test_byte_concatenation_merging


def run_all_tests():
    """Run all tokenizer tests."""
    print("=" * 80)
    print("TOKENIZER TEST SUITE")
    print("=" * 80)
    print()
    
    tests = [
        ("Common Words Not Split", test_common_words_not_split),
        ("Byte Concatenation Merging", test_byte_concatenation_merging),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*80}")
        print(f"TEST: {test_name}")
        print('='*80)
        try:
            result = test_func()
            if result:
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                failed += 1
                print(f"✗ {test_name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print(f"RESULTS: {passed} passed, {failed} failed")
    print('='*80)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

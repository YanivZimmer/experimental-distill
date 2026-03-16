#!/bin/bash
# Comprehensive test runner - Run this before training!

set -e  # Exit on error

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  SIMPLE DISTILL - COMPREHENSIVE TEST SUITE                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

run_test() {
    local test_name=$1
    local test_command=$2

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "TEST: $test_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    if eval $test_command; then
        echo -e "${GREEN}✅ PASSED: $test_name${NC}"
        ((TESTS_PASSED++))
        return 0
    else
        echo -e "${RED}❌ FAILED: $test_name${NC}"
        ((TESTS_FAILED++))
        return 1
    fi
}

# ========================================
# Phase 1: Environment Checks
# ========================================

echo "PHASE 1: Environment Validation"
echo ""

run_test "Docker installed" "docker --version > /dev/null 2>&1"
run_test "Docker Compose installed" "docker-compose --version > /dev/null 2>&1"
run_test "Docker daemon running" "docker ps > /dev/null 2>&1"

# ========================================
# Phase 2: File Validation
# ========================================

echo ""
echo "PHASE 2: File Validation"
echo ""

run_test "Dataset exists" "test -f data/train_distill.json"
run_test "Prompt template exists" "test -f baseline.txt"
run_test "Splits exist" "test -d data/splits && test -f data/splits/train.json"

# ========================================
# Phase 3: Python Tests (Local)
# ========================================

echo ""
echo "PHASE 3: Dataset Validation (Local)"
echo ""

run_test "Dataset validation" "python test_dataset.py --test-splits"

# ========================================
# Phase 4: Docker Build Tests
# ========================================

echo ""
echo "PHASE 4: Docker Build Validation"
echo ""

echo -e "${YELLOW}Building Docker image for Mac M5 Pro...${NC}"
run_test "Docker build (Mac)" "docker-compose build test"

# ========================================
# Phase 5: Docker Runtime Tests
# ========================================

echo ""
echo "PHASE 5: Docker Runtime Tests"
echo ""

run_test "Dataset validation (Docker)" "docker-compose run --rm test"

# ========================================
# Summary
# ========================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "TEST SUMMARY"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Tests Passed: ${GREEN}${TESTS_PASSED}${NC}"
echo "Tests Failed: ${RED}${TESTS_FAILED}${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║  ✅ ALL TESTS PASSED - SAFE TO TRAIN                        ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Train on Mac M5 (testing):  docker-compose up train-mac"
    echo "  2. Train on cloud GPU (prod):  bash vertex_ai_submit.sh"
    echo ""
    exit 0
else
    echo -e "${RED}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${RED}║  ❌ TESTS FAILED - FIX ERRORS BEFORE TRAINING               ║${NC}"
    echo -e "${RED}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo "Fix the errors above and run this script again."
    echo ""
    exit 1
fi

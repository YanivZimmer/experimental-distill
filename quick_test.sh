#!/bin/bash
# Quick pipeline test - completes in 30 seconds, minimal resources

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  QUICK PIPELINE TEST (No Heavy Computation)                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "This test validates your code WITHOUT:"
echo "  • Loading large models"
echo "  • Using GPU/CPU for training"
echo "  • Allocating lots of memory"
echo ""
echo "It WILL validate:"
echo "  • Dataset format is correct"
echo "  • Code logic works"
echo "  • Pipeline flow is sound"
echo ""
echo "Starting test..."
echo ""

# Run lightweight test
python test_pipeline.py

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo ""
    echo "✅ Quick test passed! Your pipeline is ready."
    echo ""
    echo "To run actual training (slow, uses resources):"
    echo "  docker-compose up train-mac"
else
    echo ""
    echo "❌ Quick test failed. Fix errors above."
fi

exit $exit_code

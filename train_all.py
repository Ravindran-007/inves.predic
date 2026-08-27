import subprocess
import sys
SYMBOLS = ['NVDA', 'AAPL', 'TSLA', 'MSFT', 'GOOGL', 'AMZN']
PERIOD = '2y'
for symbol in SYMBOLS:
    print(f'\n{'=' * 50}\nTraining {symbol}...\n{'=' * 50}')
    result = subprocess.run([sys.executable, 'train.py', '--symbols', symbol, '--period', PERIOD])
    if result.returncode != 0:
        print(f'WARNING: {symbol} training exited with code {result.returncode}')
import sys
from predict import main
if __name__ == '__main__':
    sys.argv.insert(1, '--mode'); sys.argv.insert(2, 'generate'); main()

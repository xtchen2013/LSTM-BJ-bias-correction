import subprocess

LSTMs = ["python 3.1.simple_LSTM_v6m.py",
         "python 3.2.simple_LSTM_v6m_optimized.py"
         ]


def main():
    for i in LSTMs:
        subprocess.call(i, shell=True)


if __name__ == '__main__':
    main()

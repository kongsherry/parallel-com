#include <mpi.h>
#include "PCFG.h"
#include <chrono>
#include <fstream>
#include "md5.h"
#include <iomanip>
#include <unordered_set>
using namespace std;

// 编译指令如下
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe -O1
// g++ main.cpp train.cpp guessing.cpp md5.cpp -o test.exe -O2
// mpic++ main.cpp train.cpp guessing.cpp md5.cpp -o main

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double time_hash = 0;
    double time_guess = 0;
    double time_train = 0;
    PriorityQueue q;

    double start_train = MPI_Wtime();
    q.m.train("/guessdata/Rockyou-singleLined-full.txt");
    q.m.order();
    double end_train = MPI_Wtime();
    time_train = end_train - start_train;

    // 只有rank 0加载测试数据
    unordered_set<std::string> test_set;
    int test_count = 0;
    if (rank == 0) {
        ifstream test_data("/guessdata/Rockyou-singleLined-full.txt");
        string pw;
        while (test_data >> pw)
        {
            test_count += 1;
            test_set.insert(pw);
            if (test_count >= 1000000)
            {
                break;
            }
        }
    }

    int cracked = 0;
    q.init();
    
    if (rank == 0) {
        cout << "here" << endl;
    }

    int curr_num = 0;
    double start = MPI_Wtime();
    int history = 0;
    bool should_continue = true;

    while (should_continue)
    {
        bool has_work = !q.priority.empty();
        
        // 广播是否有工作
        MPI_Bcast(&has_work, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
        
        if (!has_work) {
            should_continue = false;
            break;
        }

        q.PopNext();
        q.total_guesses = q.guesses.size();
        
        if (rank == 0) {
            if (q.total_guesses - curr_num >= 100000)
            {
                cout << "Guesses generated: " << history + q.total_guesses << endl;
                curr_num = q.total_guesses;

                int generate_n = 10000000;
                if (history + q.total_guesses > generate_n)
                {
                    double end = MPI_Wtime();
                    time_guess = end - start;
                    cout << "Guess time:" << time_guess - time_hash << "seconds" << endl;
                    cout << "Hash time:" << time_hash << "seconds" << endl;
                    cout << "Train time:" << time_train << "seconds" << endl;
                    cout << "Cracked:" << cracked << endl;
                    should_continue = false;
                }
            }

            if (curr_num > 1000000)
            {
                double start_hash = MPI_Wtime();
                bit32 state[4];
                for (string pw : q.guesses)
                {
                    if (test_set.find(pw) != test_set.end()) {
                        cracked += 1;
                    }
                    MD5Hash(pw, state);
                }

                double end_hash = MPI_Wtime();
                time_hash += (end_hash - start_hash);

                history += curr_num;
                curr_num = 0;
                q.guesses.clear();
            }
        }
        
        // 广播是否继续
        MPI_Bcast(&should_continue, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);
    }

    // 确保所有进程同步
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}

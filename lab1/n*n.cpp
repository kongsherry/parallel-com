#include <iostream>
#include <windows.h>
using namespace std;
const int N = 3000;
int a[N];
int b[N][N];
int sum[N];
int step = 1000;

void init()
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            b[i][j] = i + j;
        }
        a[i] = i;
    }
}

void ordinary()
{
    long long int head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int l = 0; l < step; l++)
    {
        for (int i = 0; i < N; i++)
        {
            sum[i] = 0;
            for (int j = 0; j < N; j++)
                sum[i] += a[j] * b[j][i];
        }
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "ordinary:" << (tail - head) * 1000.0 / freq/step << "ms" << endl;
}

void far_ordinary()
{
    long long int head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int l = 0; l < step; l++)
    {
        for (int i = 0; i < N; i++)
            sum[i] = 0;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++)
                sum[i] += a[j] * b[j][i];
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "far_ordinary:" << (tail - head) * 1000.0 / freq / step << "ms" << endl;
}

int main()
{
    init();
    ordinary();
    far_ordinary();
    return 0;
}

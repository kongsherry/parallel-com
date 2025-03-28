#include <iostream>
#include <windows.h>
#include<cmath>
using namespace std;
int n = 30;
const long long int N = pow(2, n);
long long int* c = new long long int[N];
int step = 10;

void init()
{
    for (int i = 0; i < N; i++) {
        c[i] = i;
    }
}

void ordinary()
{
    long long int head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    long long int sum = 0;
    for (int l = 0; l < step; l++)
    {
        for (int i = 0; i < N; i++)
            sum += c[i];
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "ordinary:" << (tail - head) * 1000.0 / freq / step << "ms" << endl;
}

void f_ordinary()
{
    long long int head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    long long int sum = 0, sum1 = 0, sum2 = 0;
    for (int l = 0; l < step; l++)
    {
        for (long long int i = 0; i < N; i += 2)
        {
            sum1 += c[i];
            sum2 += c[i + 1];
        }
        sum = sum1 + sum2;
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "f_ordinary:" << (tail - head) * 1000.0 / freq / step << "ms" << endl;
}
void re(int n)
{
    if (n == 1)
        return;
    else
    {
        for (int i = 0; i < n / 2; i++)
        {
            c[i] += c[n - i - 1];
        }
        n = n / 2;
        re(n);
    }
}

void f_f_ordinary()
{
    long long int head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER*)&freq);
    QueryPerformanceCounter((LARGE_INTEGER*)&head);
    for (int l = 0; l < step; l++)
    {
        re(N);
    }
    QueryPerformanceCounter((LARGE_INTEGER*)&tail);
    cout << "f_f_ordinary:" << (tail - head) * 1000.0 / freq / step << "ms" << endl;
}



int main()
{
    init();
    ordinary();
    f_ordinary();
    f_f_ordinary();
    return 0;
}

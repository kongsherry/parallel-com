#ifndef MD5_H_INCLUDED
#define MD5_H_INCLUDED
#include <iostream>
#include <string>
#include <cstring>


using namespace std;

// 定义了Byte，便于使用
typedef unsigned char Byte;
// 定义了32比特
typedef unsigned int bit32;

// MD5的一系列参数。参数是固定的，其实你不需要看懂这些
#define s11 7
#define s12 12
#define s13 17
#define s14 22
#define s21 5
#define s22 9
#define s23 14
#define s24 20
#define s31 4
#define s32 11
#define s33 16
#define s34 23
#define s41 6
#define s42 10
#define s43 15
#define s44 21
#include <emmintrin.h>
#include <tmmintrin.h>

/**
 * @Basic MD5 functions.
 *
 * @param there bit32.
 *
 * @return one bit32.
 */
// 定义了一系列MD5中的具体函数
// 这四个计算函数是需要你进行SIMD并行化的
// 可以看到，FGHI四个函数都涉及一系列位运算，在数据上是对齐的，非常容易实现SIMD的并行化

#define F(x, y, z) (((x) & (y)) | ((~x) & (z)))
#define G(x, y, z) (((x) & (z)) | ((y) & (~z)))
#define H(x, y, z) ((x) ^ (y) ^ (z))
#define I(x, y, z) ((y) ^ ((x) | (~z)))

/**
 * @Rotate Left.
 *
 * @param {num} the raw number.
 *
 * @param {n} rotate left n.
 *
 * @return the number after rotated left.
 */
// 定义了一系列MD5中的具体函数
// 这五个计算函数（ROTATELEFT/FF/GG/HH/II）和之前的FGHI一样，都是需要你进行SIMD并行化的
// 但是你需要注意的是#define的功能及其效果，可以发现这里的FGHI是没有返回值的，为什么呢？你可以查询#define的含义和用法
#define ROTATELEFT(num, n) (((num) << (n)) | ((num) >> (32-(n))))

#define FF(a, b, c, d, x, s, ac) { \
  (a) += F ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

#define GG(a, b, c, d, x, s, ac) { \
  (a) += G ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}
#define HH(a, b, c, d, x, s, ac) { \
  (a) += H ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}
#define II(a, b, c, d, x, s, ac) { \
  (a) += I ((b), (c), (d)) + (x) + ac; \
  (a) = ROTATELEFT ((a), (s)); \
  (a) += (b); \
}

void MD5Hash(string input, bit32 *state);

inline __m128i F_sse(__m128i x, __m128i y, __m128i z) {
    __m128i xy = _mm_and_si128(x, y);
    __m128i notx = _mm_andnot_si128(x, _mm_set1_epi32(0xFFFFFFFF));
    __m128i notx_z = _mm_and_si128(notx, z);
    return _mm_or_si128(xy, notx_z);
}

inline __m128i G_sse(__m128i x, __m128i y, __m128i z) {
    __m128i xz = _mm_and_si128(x, z);
    __m128i notz = _mm_andnot_si128(z, _mm_set1_epi32(0xFFFFFFFF));
    __m128i y_notz = _mm_and_si128(y, notz);
    return _mm_or_si128(xz, y_notz);
}

inline __m128i H_sse(__m128i x, __m128i y, __m128i z) {
    __m128i xy = _mm_xor_si128(x, y);
    return _mm_xor_si128(xy, z);
}

inline __m128i I_sse(__m128i x, __m128i y, __m128i z) {
    __m128i notz = _mm_andnot_si128(z, _mm_set1_epi32(0xFFFFFFFF));
    __m128i x_or_notz = _mm_or_si128(x, notz);
    return _mm_xor_si128(y, x_or_notz);
}

inline __m128i ROTATELEFT_sse(__m128i num, int n) {
    return _mm_or_si128(_mm_slli_epi32(num, n), _mm_srli_epi32(num, 32 - n));
}

// SSE版本的FF/GG/HH/II
#define FF_SSE(a, b, c, d, x, s, ac) { \
    __m128i temp = F_sse(b, c, d); \
    temp = _mm_add_epi32(temp, x); \
    temp = _mm_add_epi32(temp, _mm_set1_epi32(ac)); \
    a = _mm_add_epi32(a, temp); \
    a = ROTATELEFT_sse(a, s); \
    a = _mm_add_epi32(a, b); \
}

#define GG_SSE(a, b, c, d, x, s, ac) { \
    __m128i temp = G_sse(b, c, d); \
    temp = _mm_add_epi32(temp, x); \
    temp = _mm_add_epi32(temp, _mm_set1_epi32(ac)); \
    a = _mm_add_epi32(a, temp); \
    a = ROTATELEFT_sse(a, s); \
    a = _mm_add_epi32(a, b); \
}

#define HH_SSE(a, b, c, d, x, s, ac) { \
    __m128i temp = H_sse(b, c, d); \
    temp = _mm_add_epi32(temp, x); \
    temp = _mm_add_epi32(temp, _mm_set1_epi32(ac)); \
    a = _mm_add_epi32(a, temp); \
    a = ROTATELEFT_sse(a, s); \
    a = _mm_add_epi32(a, b); \
}

#define II_SSE(a, b, c, d, x, s, ac) { \
    __m128i temp = I_sse(b, c, d); \
    temp = _mm_add_epi32(temp, x); \
    temp = _mm_add_epi32(temp, _mm_set1_epi32(ac)); \
    a = _mm_add_epi32(a, temp); \
    a = ROTATELEFT_sse(a, s); \
    a = _mm_add_epi32(a, b); \
}

void MD5Hash_SIMD(string input[4], __m128i* state);



#endif // MD5_H_INCLUDED

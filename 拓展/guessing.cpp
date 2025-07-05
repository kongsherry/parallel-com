#include "PCFG.h"
#include <mpi.h>
#include <cstring>
#include <sstream>
#include <vector>

using namespace std;

// 辅助函数：序列化PT对象
vector<char> SerializePT(const PT& pt) {
    vector<char> buffer;
    
    // 序列化content
    int content_size = pt.content.size();
    buffer.insert(buffer.end(), (char*)&content_size, (char*)&content_size + sizeof(int));
    
    for (const segment& seg : pt.content) {
        // 序列化segment
        buffer.insert(buffer.end(), (char*)&seg.type, (char*)&seg.type + sizeof(int));
        buffer.insert(buffer.end(), (char*)&seg.length, (char*)&seg.length + sizeof(int));
    }
    
    // 序列化其他成员
    buffer.insert(buffer.end(), (char*)&pt.pivot, (char*)&pt.pivot + sizeof(int));
    buffer.insert(buffer.end(), (char*)&pt.preterm_prob, (char*)&pt.preterm_prob + sizeof(float));
    buffer.insert(buffer.end(), (char*)&pt.prob, (char*)&pt.prob + sizeof(float));
    
    // 序列化indices
    int indices_size = pt.curr_indices.size();
    buffer.insert(buffer.end(), (char*)&indices_size, (char*)&indices_size + sizeof(int));
    buffer.insert(buffer.end(), (char*)pt.curr_indices.data(), 
                 (char*)pt.curr_indices.data() + indices_size * sizeof(int));
    
    int max_indices_size = pt.max_indices.size();
    buffer.insert(buffer.end(), (char*)&max_indices_size, (char*)&max_indices_size + sizeof(int));
    buffer.insert(buffer.end(), (char*)pt.max_indices.data(), 
                 (char*)pt.max_indices.data() + max_indices_size * sizeof(int));
    
    return buffer;
}

// 辅助函数：反序列化PT对象
PT DeserializePT(const vector<char>& buffer, const model& m) {
    PT pt;
    size_t pos = 0;
    
    // 反序列化content
    int content_size;
    memcpy(&content_size, &buffer[pos], sizeof(int));
    pos += sizeof(int);
    
    for (int i = 0; i < content_size; i++) {
        int type, length;
        memcpy(&type, &buffer[pos], sizeof(int));
        pos += sizeof(int);
        memcpy(&length, &buffer[pos], sizeof(int));
        pos += sizeof(int);
        
        // 根据类型查找模型中的segment
        segment seg(type, length);
        pt.content.push_back(seg);
    }
    
    // 反序列化其他成员
    memcpy(&pt.pivot, &buffer[pos], sizeof(int));
    pos += sizeof(int);
    memcpy(&pt.preterm_prob, &buffer[pos], sizeof(float));
    pos += sizeof(float);
    memcpy(&pt.prob, &buffer[pos], sizeof(float));
    pos += sizeof(float);
    
    // 反序列化indices
    int indices_size;
    memcpy(&indices_size, &buffer[pos], sizeof(int));
    pos += sizeof(int);
    pt.curr_indices.resize(indices_size);
    memcpy(pt.curr_indices.data(), &buffer[pos], indices_size * sizeof(int));
    pos += indices_size * sizeof(int);
    
    int max_indices_size;
    memcpy(&max_indices_size, &buffer[pos], sizeof(int));
    pos += sizeof(int);
    pt.max_indices.resize(max_indices_size);
    memcpy(pt.max_indices.data(), &buffer[pos], max_indices_size * sizeof(int));
    pos += max_indices_size * sizeof(int);
    
    return pt;
}

void PriorityQueue::CalProb(PT &pt) {
    pt.prob = pt.preterm_prob;
    int index = 0;

    for (int idx : pt.curr_indices) {
        if (pt.content[index].type == 1) {
            pt.prob *= m.letters[m.FindLetter(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.letters[m.FindLetter(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 2) {
            pt.prob *= m.digits[m.FindDigit(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.digits[m.FindDigit(pt.content[index])].total_freq;
        }
        if (pt.content[index].type == 3) {
            pt.prob *= m.symbols[m.FindSymbol(pt.content[index])].ordered_freqs[idx];
            pt.prob /= m.symbols[m.FindSymbol(pt.content[index])].total_freq;
        }
        index += 1;
    }
}

void PriorityQueue::init() {
    for (PT pt : m.ordered_pts) {
        for (segment seg : pt.content) {
            if (seg.type == 1) {
                pt.max_indices.emplace_back(m.letters[m.FindLetter(seg)].ordered_values.size());
            }
            if (seg.type == 2) {
                pt.max_indices.emplace_back(m.digits[m.FindDigit(seg)].ordered_values.size());
            }
            if (seg.type == 3) {
                pt.max_indices.emplace_back(m.symbols[m.FindSymbol(seg)].ordered_values.size());
            }
        }
        pt.preterm_prob = float(m.preterm_freq[m.FindPT(pt)]) / m.total_preterm;
        CalProb(pt);
        priority.emplace_back(pt);
    }
}

void PriorityQueue::PopMultiple(int batch_size) {
    int actual_batch = min(batch_size, (int)priority.size());
    vector<PT> batch(priority.begin(), priority.begin() + actual_batch);
    priority.erase(priority.begin(), priority.begin() + actual_batch);
    
    GenerateBatch(batch);
}

void PriorityQueue::GenerateBatch(const vector<PT>& batch) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        // 主进程分配任务
        vector<vector<PT>> assignments(size);
        for (int i = 0; i < batch.size(); i++) {
            assignments[i % size].push_back(batch[i]);
        }

        // 发送任务给从进程
        for (int dest = 1; dest < size; dest++) {
            int count = assignments[dest].size();
            MPI_Send(&count, 1, MPI_INT, dest, 0, MPI_COMM_WORLD);
            
            for (PT pt : assignments[dest]) {
                vector<char> buffer = SerializePT(pt);
                int buffer_size = buffer.size();
                MPI_Send(&buffer_size, 1, MPI_INT, dest, 1, MPI_COMM_WORLD);
                MPI_Send(buffer.data(), buffer_size, MPI_CHAR, dest, 2, MPI_COMM_WORLD);
            }
        }

        // 主进程处理自己的任务
        vector<string> local_guesses;
        for (PT pt : assignments[0]) {
            vector<string> guesses = GenerateForPT(pt);
            local_guesses.insert(local_guesses.end(), guesses.begin(), guesses.end());
        }
        guesses.insert(guesses.end(), local_guesses.begin(), local_guesses.end());
        total_guesses += local_guesses.size();

        // 接收从进程的结果
        for (int src = 1; src < size; src++) {
            int guess_count;
            MPI_Recv(&guess_count, 1, MPI_INT, src, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            if (guess_count > 0) {
                int total_chars;
                MPI_Recv(&total_chars, 1, MPI_INT, src, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                vector<char> buffer(total_chars);
                MPI_Recv(buffer.data(), total_chars, MPI_CHAR, src, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // 解析猜测
                int pos = 0;
                for (int i = 0; i < guess_count; i++) {
                    int len;
                    memcpy(&len, &buffer[pos], sizeof(int));
                    pos += sizeof(int);
                    string guess(&buffer[pos], len);
                    pos += len;
                    guesses.push_back(guess);
                }
                total_guesses += guess_count;
            }
        }
    } else {
        // 从进程接收任务
        int count;
        MPI_Recv(&count, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        vector<PT> local_pts;
        for (int i = 0; i < count; i++) {
            int buffer_size;
            MPI_Recv(&buffer_size, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            vector<char> buffer(buffer_size);
            MPI_Recv(buffer.data(), buffer_size, MPI_CHAR, 0, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            local_pts.push_back(DeserializePT(buffer, m));
        }

        // 处理任务
        vector<string> local_guesses;
        for (PT pt : local_pts) {
            vector<string> guesses = GenerateForPT(pt);
            local_guesses.insert(local_guesses.end(), guesses.begin(), guesses.end());
        }

        // 发送结果回主进程
        int guess_count = local_guesses.size();
        MPI_Send(&guess_count, 1, MPI_INT, 0, 3, MPI_COMM_WORLD);
        
        if (guess_count > 0) {
            // 计算总字符数
            int total_chars = 0;
            for (const string& s : local_guesses) {
                total_chars += sizeof(int) + s.size();
            }
            
            // 打包数据
            vector<char> buffer(total_chars);
            int pos = 0;
            for (const string& s : local_guesses) {
                int len = s.size();
                memcpy(&buffer[pos], &len, sizeof(int));
                pos += sizeof(int);
                memcpy(&buffer[pos], s.data(), len);
                pos += len;
            }
            
            MPI_Send(&total_chars, 1, MPI_INT, 0, 4, MPI_COMM_WORLD);
            MPI_Send(buffer.data(), total_chars, MPI_CHAR, 0, 5, MPI_COMM_WORLD);
        }
    }
}

vector<string> PriorityQueue::GenerateForPT(PT pt) {
    vector<string> res;
    
    if (pt.content.size() == 1) {
        segment* a;
        if (pt.content[0].type == 1) {
            a = &m.letters[m.FindLetter(pt.content[0])];
        } else if (pt.content[0].type == 2) {
            a = &m.digits[m.FindDigit(pt.content[0])];
        } else if (pt.content[0].type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content[0])];
        }
        
        for (int i = 0; i < pt.max_indices[0]; i++) {
            res.push_back(a->ordered_values[i]);
        }
    } else {
        string guess;
        int seg_idx = 0;
        for (int idx : pt.curr_indices) {
            if (pt.content[seg_idx].type == 1) {
                guess += m.letters[m.FindLetter(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 2) {
                guess += m.digits[m.FindDigit(pt.content[seg_idx])].ordered_values[idx];
            } else if (pt.content[seg_idx].type == 3) {
                guess += m.symbols[m.FindSymbol(pt.content[seg_idx])].ordered_values[idx];
            }
            seg_idx += 1;
            if (seg_idx == pt.content.size() - 1) {
                break;
            }
        }

        segment* a;
        if (pt.content.back().type == 1) {
            a = &m.letters[m.FindLetter(pt.content.back())];
        } else if (pt.content.back().type == 2) {
            a = &m.digits[m.FindDigit(pt.content.back())];
        } else if (pt.content.back().type == 3) {
            a = &m.symbols[m.FindSymbol(pt.content.back())];
        }

        for (int i = 0; i < pt.max_indices.back(); i++) {
            res.push_back(guess + a->ordered_values[i]);
        }
    }
    
    return res;
}

vector<PT> PT::NewPTs() {
    vector<PT> res;
    if (content.size() == 1) {
        return res;
    } else {
        int init_pivot = pivot;
        for (int i = pivot; i < curr_indices.size() - 1; i++) {
            curr_indices[i] += 1;
            if (curr_indices[i] < max_indices[i]) {
                pivot = i;
                res.emplace_back(*this);
            }
            curr_indices[i] -= 1;
        }
        pivot = init_pivot;
        return res;
    }
}

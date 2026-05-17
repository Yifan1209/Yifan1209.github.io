---
title: Advanced C++ Notes
date: 2024-01-01
categories:
  - Coding
tags:
  - C++
  - STL
description: C++ STL containers, lambda expressions, const pointers, virtual functions, and interview coding examples.
---

## STL Containers

### set

```cpp
set<int> val = {6, 10, 5, 1};
val.insert(1);
val.erase(1);
```

### unordered_map

```cpp
unordered_map<string, int> umap;
umap["foo"] = 10;
umap["bar"] = 20;

for (auto& [k, v] : umap)
    cout << k << " " << v << endl;
```

## Lambda Expression

```cpp
sort(words.begin(), words.end(),
     [](const string& a, const string& b){ return a.size() < b.size(); });
```

需要 `#include <algorithm>`

## String

### substr

两个参数是起始位置和**长度**（不是终止位置）：

```cpp
string s = "iiyokoiyo";
string s1 = s.substr(1, 4);  // "iyok"（从1开始，长度4）
string s2 = s.substr(1);     // "iyokoiyo"（从1到结尾）
```

## const

```cpp
const int* ptr = &i1;   // ptr本身可变，*ptr不可变
int* const ptr2 = &i3;  // ptr2本身不可变，*ptr2可变
```

## virtual

C++中要在子类中override函数，父类必须用 `virtual` 声明：

```cpp
class Animal {
public:
    virtual void speak() { cout << "..." << endl; }
};

class Dog : public Animal {
public:
    void speak() override { cout << "Woof!" << endl; }
};

int main() {
    Animal* a = new Dog();
    a->speak();  // 输出 "Woof!"，而非 "..."
    // 如果父类没有 virtual，则输出 "..."（静态绑定）
}
```

没有 `virtual` 时，通过基类指针调用会走静态绑定，无法实现多态。

## 面试题：Cheapest Flights Within K Stops

```cpp
int findCheapestPrice(int n, vector<vector<int>>& flights, int src, int dst, int k) {
    vector<vector<int>> prices(n, vector<int>(n, INT_MAX));
    for (auto& f : flights)
        prices[f[0]][f[1]] = f[2];

    // BFS with (city, stops, cost)
    queue<tuple<int,int,int>> q;
    q.push({src, 0, 0});
    int ans = INT_MAX;

    while (!q.empty()) {
        auto [city, stops, cost] = q.front(); q.pop();
        if (city == dst) { ans = min(ans, cost); continue; }
        if (stops > k) continue;
        for (int next = 0; next < n; next++) {
            if (prices[city][next] != INT_MAX)
                q.push({next, stops+1, cost + prices[city][next]});
        }
    }
    return ans == INT_MAX ? -1 : ans;
}
```

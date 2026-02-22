#include <iostream>
#include <vector>
#include <string>
#include <algorithm>

typedef std::vector<int> B;

B bfs(const std::string &s)
{
    B d;
    for (int i = (int)s.size() - 1; i >= 0; --i)
        d.push_back(s[i] - '0');
    while (d.size() > 1 and d.back() == 0)
        d.pop_back();
    return d;
}

void bt(B &d)
{
    while (d.size() > 1 and d.back() == 0)
        d.pop_back();
}

bool bz(const B &d)
{
    return d.size() == 1 and d[0] == 0;
}

std::string bs(const B &d)
{
    std::string s;
    for (int i = (int)d.size() - 1; i >= 0; --i)
        s.push_back((char)('0' + d[i]));
    return s;
}

int bc(const B &d, int x)
{
    B t;
    if (x == 0)
        t.push_back(0);
    while (x > 0)
    {
        t.push_back(x % 10);
        x /= 10;
    }
    if (d.size() != t.size())
        return (d.size() < t.size() ? -1 : 1);
    for (int i = (int)d.size() - 1; i >= 0; --i)
    {
        if (d[i] != t[i])
            return (d[i] < t[i] ? -1 : 1);
    }
    return 0;
}

int bm(const B &d, int m)
{
    int r = 0;
    for (int i = (int)d.size() - 1; i >= 0; --i)
        r = (r * 10 + d[i]) % m;
    return r;
}

void bsub(B &d, int x)
{
    int i = 0;
    while (x > 0 or i < (int)d.size())
    {
        int s = x % 10;
        x /= 10;
        d[i] -= s;
        if (d[i] < 0)
        {
            d[i] += 10;
            d[i + 1]--;
        }
        ++i;
    }
    for (int j = 0; j + 1 < (int)d.size(); ++j)
    {
        if (d[j] < 0)
        {
            d[j] += 10;
            d[j + 1]--;
        }
    }
    bt(d);
}

B bd(const B &d, int m)
{
    B q(d.size(), 0);
    int r = 0;
    for (int i = (int)d.size() - 1; i >= 0; --i)
    {
        int c = r * 10 + d[i];
        q[i] = c / m;
        r = c % m;
    }
    bt(q);
    return q;
}

int cv(char c)
{
    if (c >= '0' and c <= '9')
        return c - '0';
    return 10 + (c - 'A');
}

int gb;
std::vector<int> av;
std::vector<char> ac;
std::vector<std::string> mk;
std::vector<int> mc;
std::vector<std::vector<std::string>> me;

int solve(const B &x)
{
    std::string key = bs(x);

    for (int i = 0; i < (int)mk.size(); ++i)
        if (mk[i] == key)
            return i;

    int id = (int)mk.size();
    mk.push_back(key);
    mc.push_back(0);
    me.push_back(std::vector<std::string>());

    if (bz(x))
    {
        mc[id] = 1;
        me[id].push_back("");
        return id;
    }

    for (int i = 0; i < (int)av.size(); ++i)
    {
        int d = av[i];
        char ch = ac[i];

        if (bc(x, d) < 0)
            continue;

        B y = x;
        bsub(y, d);
        if (bm(y, gb) != 0)
            continue;

        B t = bd(y, gb);

        int ni = solve(t);
        if (mc[ni] == 0)
            continue;

        for (int k = 0; k < (int)me[ni].size(); ++k)
        {
            const std::string &sf = me[ni][k];

            if (sf.empty() and d == 0)
                continue;

            std::string cd;
            cd.reserve(1 + sf.size());
            cd.push_back(ch);
            cd += sf;

            bool dup = false;
            for (int z = 0; z < (int)me[id].size(); ++z)
            {
                if (me[id][z] == cd)
                {
                    dup = true;
                    break;
                }
            }
            if (!dup)
            {
                me[id].push_back(cd);
                if (mc[id] < 2)
                    ++mc[id];
                if (me[id].size() > 2)
                    me[id].resize(2);
            }

            if (mc[id] >= 2 and me[id].size() >= 2)
                break;
        }
        if (mc[id] >= 2 and me[id].size() >= 2)
            break;
    }

    return id;
}

int main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    std::cin >> gb;
    std::string al;
    std::cin >> al;
    std::string nd;
    std::cin >> nd;

    for (int i = 0; i < (int)al.size(); ++i)
    {
        ac.push_back(al[i]);
        av.push_back(cv(al[i]));
    }

    B n = bfs(nd);
    int ai = solve(n);

    std::vector<std::string> r;
    for (int i = 0; i < (int)me[ai].size(); ++i)
        if (!me[ai][i].empty())
            r.push_back(me[ai][i]);

    if (mc[ai] == 0 or r.empty())
    {
        std::cout << "Impossible";
        return 0;
    }

    for (int i = 0; i < (int)r.size(); ++i)
        std::reverse(r[i].begin(), r[i].end());

    if (mc[ai] == 1)
    {
        std::cout << "Unique\n";
        std::cout << r[0];
    }
    else
    {
        std::cout << "Ambiguous\n";
        std::cout << r[0] << '\n';
        std::cout << r[1];
    }

    return 0;
}
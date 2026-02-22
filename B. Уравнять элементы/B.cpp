#include <iostream>
#include <vector>
#include <utility>

struct Op
{
    char type;
    int a, b, c;
};

int main()
{
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    int n, m;
    std::cin >> n >> m;

    std::vector<std::vector<int>> a(n, std::vector<int>(m));
    for (size_t i = 0; i < n; ++i)
    {
        for (size_t j = 0; j < m; ++j)
            std::cin >> a[i][j];
    }

    std::vector<Op> ops;
    std::vector<int> rowValue(n, -1);
    std::vector<std::pair<int, int>> rowPair(n);

    for (int i = 0; i < n; ++i)
    {
        std::vector<int> first(101, -1);

        bool found = false;
        for (int j = 0; j < m; ++j)
        {
            int x = a[i][j];
            if (first[x] == -1)
            {
                first[x] = j;
            }
            else
            {
                rowValue[i] = x;
                rowPair[i] = std::make_pair(first[x] + 1, j + 1);
                ops.push_back(Op{'R', i + 1, first[x] + 1, j + 1});
                found = true;
                break;
            }
        }

        if (!found)
            return 0;
    }

    std::vector<int> firstRowWithVal(101, -1);
    int r1 = -1, r2 = -1, target = -1;

    for (size_t i = 0; i < n; ++i)
    {
        int x = rowValue[i];
        if (firstRowWithVal[x] == -1)
            firstRowWithVal[x] = i;
        else
        {
            r1 = firstRowWithVal[x];
            r2 = i;
            target = x;
            break;
        }
    }

    for (int j = 0; j < m; j++)
        ops.push_back(Op{'C', j + 1, r1 + 1, r2 + 1});

    std::cout << ops.size() << '\n';
    for (std::size_t k = 0; k < ops.size(); ++k)
        std::cout << ops[k].type << " " << ops[k].a << " " << ops[k].b << " " << ops[k].c << '\n';

    return 0;
}
class UnionFind:
    def __init__(self, size):
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return

        if self.rank[root_x] < self.rank[root_y]:
            self.parent[root_x] = root_y
        elif self.rank[root_x] > self.rank[root_y]:
            self.parent[root_y] = root_x
        else:
            self.parent[root_y] = root_x
            self.rank[root_x] += 1


def test_union_find():
    size = 10
    uf = UnionFind(size)

    # Perform union operations
    uf.union(0, 2)
    uf.union(4, 2)
    uf.union(3, 8)
    uf.union(6, 5)

    # Check find operations
    assert uf.find(0) == 4
    assert uf.find(2) == 4
    assert uf.find(4) == 4
    assert uf.find(3) == 3
    assert uf.find(8) == 3
    assert uf.find(6) == 6
    assert uf.find(5) == 6

    # Perform more union operations
    uf.union(1, 7)
    uf.union(1, 9)
    uf.union(5, 9)

    # Check updated find operations
    assert uf.find(1) == 6
    assert uf.find(7) == 6
    assert uf.find(9) == 6
    assert uf.find(5) == 6

    print("All tests passed successfully.")


# Run the test
test_union_find()

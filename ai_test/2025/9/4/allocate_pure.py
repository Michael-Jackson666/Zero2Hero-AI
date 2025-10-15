import heapq

'''无调包版本'''
def allocate_pure(m: int, n: int, sample: list[int]):
    if m == 0 or n <= 0 or not sample:
        print(0)
        return

    tasks = sorted(sample[:m], reverse=True)
    heap = [(0, i) for i in range(n)]
    heapq.heapify(heap)

    ans = 0
    for task in tasks:
        load, idx = heapq.heappop(heap)
        load += task
        ans = max(ans, load)
        heapq.heappush(heap, (load, idx))

    print(ans)


if __name__ == '__main__':
    n = int(input().strip())
    m = int(input().strip())
    sample = list(map(int, input().strip().split()))
    allocate_pure(m, n, sample)
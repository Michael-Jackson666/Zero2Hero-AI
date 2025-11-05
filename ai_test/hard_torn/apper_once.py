from collections import Counter
class Solution:
    def apper_once(self, nums: int) -> int:
        """
        给一个整数数组，数组中只有一个数字出现一次，其他数字都出现两次，返回只出现一次的数字
        """
        count = Counter(nums)
        for num in count:
            if count[num] == 1:
                return num
        
def test():
    sol = Solution()
    nums1 = [4,3,2,7,8,2,3,1,4]
    nums2 = [1,2,3,4,3,2,1]
    nums3 = [7,8,9,8,7]
    result1 = sol.apper_once(nums1)
    result2 = sol.apper_once(nums2)
    result3 = sol.apper_once(nums3)
    print(result1)
    print(result2)
    print(result3)

test()
import math
class Solution:
    def mininum(self, nums:int) -> int:
        """
        给一个整数数组(有正数，有负数),返回任意两数相加的绝对值的最小值
        """
        n = len(nums)
        ans = math.inf
        for i in range(n):
            for j in range(n):
                if i != j:
                    current_sum = abs(nums[i] - nums[j])
                    ans = min(current_sum, ans)
        return ans
    
def test():
    sol = Solution()
    nums1 = [1,2,3,-23,24,4,-9]
    nums2 = [23,342,432,-234,-431,428]
    nums3 = [-2334,324,543,643,-356,290]
    result1 = sol.mininum(nums1)
    result2 = sol.mininum(nums2)
    result3 = sol.mininum(nums3)
    print(result1)
    print(result2)
    print(result3)
    
test()

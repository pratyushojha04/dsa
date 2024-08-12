def findCircleNum(isConnected):
        if not isConnected:
            return 0
        n=len(isConnected)
        visit=[False]*n
        def dfs (u):
            for v in range(n):
                if isConnected[u][v] == 1 and visit[v] == False:
                    visit[v] = True
                    dfs(v)
        count = 0
        for i in range(n):
            if visit[i] ==False:
                count+=1
                visit[i] = True
                dfs(i)
        return count


    
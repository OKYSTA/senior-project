# DBSCANクラスタリング
def DBSCAN(data_point_Set , eps , minPts):
    coreObjs = {}
    C = {}
    n = len(data_point_Set)
    for k_id,v_point in data_point_Set.items():
        neibor = get_neibor(v_point , data_point_Set , eps)
        if len(neibor)>=minPts:
            coreObjs[k_id] = neibor
    oldCoreObjs = coreObjs.copy()
    k = 0
    notAccess = list(data_point_Set.keys())
    while len(coreObjs)>0:
        OldNotAccess = []
        OldNotAccess.extend(notAccess)
        cores = list(coreObjs.keys())
        # 待ちポイントの設置 # 初期点
        core = cores[0]
        queue = []
        queue.append(core)
        notAccess.remove(core)
        while len(queue)>0:
            q = queue[0]
            del queue[0]
            if q in oldCoreObjs.keys() :
                delte = [val for val in oldCoreObjs[q] if val in notAccess]
                queue.extend(delte)
                notAccess = [val for val in notAccess if val not in delte]
        k += 1
        C[k] = [val for val in OldNotAccess if val not in notAccess]
        for x in C[k]:
            if x in coreObjs.keys():
                del coreObjs[x]
    return C
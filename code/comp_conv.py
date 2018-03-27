import pickle

fb1 = open ('Convergence_test_bigHeading1.pkl','rb')
fb2 = open('Convergence_test_bigHeading2.pkl','rb')
list1 = pickle.load(fb1)
list2 = pickle.load(fb2)

n_set=[50,100,200,400]

for i in range(len(list1)):
    print('Difference with',n_set[i],'particles:')
    lst1 = list1[i]
    lst2 = list2[i]
    for j in range(len(lst1)):
        if lst1[j] != lst2[j]:
            print(n_set[i],'particles,',int(j/10),',',j-int(j/10)*10,':',lst1[j],',',lst2[j])

#After inspection, set these by hand
list1[0][11] = 'n'
list1[1][18] = 'b'
list1[1][19] = 'b'

fb3 = open('Convergence_corrected_bigHeading.pkl','wb')
pickle.dump(list1,fb3)
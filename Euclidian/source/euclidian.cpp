#include<iostream>
#include<vector>
#include<stdlib.h>
#include<time.h>
#include<ctime>
#include<math.h>
#include <algorithm>

using namespace std;

double t0, t1;
vector<int> sizo{ 10000, 15000 ,20000 ,25000 };
vector<int> dimos{ 4, 6, 8, 10, 12, 18, 20 };
vector<int> res;

void print(vector<vector<int>> a)
{
    for (int i = 0; i < a.size(); i++) {
        for (int j = 0; j < a[i].size(); j++) {
            cout << a[i][j] << " ";
        }cout << endl;
    }
}

void print_res_highest(vector<int> a)
{

   int max = *max_element(res.begin(),res.end()) ;
   cout << "*max_element = " << max << endl;
}

void print_res_lowest(vector<int> a)
{
   int min = *min_element(res.begin(),res.end()) ;
   cout << "*min_element = " << min << endl;
}

void randomizer(vector<vector<int>>& a, int sizos, int dims)
{
    for (int i = 0; i < sizos; i++) {
        vector<int> aux;
        for (int j = 0; j < dims; j++) {
            aux.push_back(rand() % 100);
        }
        a.push_back(aux);
    }
}
float euclidianDistance(vector<int>& a, vector<int>& b)
{
    
    float tmp = 0;
    for (int i = 0; i < dimos[i]; i++) {
        tmp += pow((a[i] - b[i]), 2);
        
    }
    float aux = sqrt(tmp);
    res.push_back((int)tmp);
    return aux;
}

int main()
{
    srand(time(NULL));
    for (int i = 0; i < sizo.size(); i++) 
    {
        for (int j = 0; j < dimos.size(); j++) 
        {
            vector<vector<int>> X;
            randomizer(X, sizo[i], dimos[j]);
            //print (X);
            //cout<<"---------- "<<sizo[i]<<"---"<<dimos[j]<<endl;
            t0 = clock();
            for (int i = 0; i < X.size(); i++) 
            {
                //cout<<endl;
                for (int j = i + 1; j < X.size(); j++) 
                {
                    euclidianDistance(X[i], X[j]);
                }
            }
            t1 = clock();

            double time = (double(t1 - t0) / CLOCKS_PER_SEC);
            cout << "Execution Time size(" << sizo[i] << ") and Dim(" << dimos[j] << "):" << time << endl ;
            print_res_highest(res);
            print_res_lowest(res);
            res.clear();
        }
    }
}

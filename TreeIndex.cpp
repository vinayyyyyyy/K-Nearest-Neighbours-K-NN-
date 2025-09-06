#include "TreeIndex.h"

TreeIndex* TreeIndex::instance = nullptr;
KDTreeIndex* KDTreeIndex::instance = nullptr;
RPTreeIndex* RPTreeIndex::instance = nullptr;

TreeIndex& TreeIndex::GetInstance()
{
    if (instance == nullptr)
    {
        instance = new TreeIndex;
    }
    return *instance;
}
//Implementation of KDTreeIndex functions

//Implementing a static method which generates new instance of an index
KDTreeIndex& KDTreeIndex::GetInstance()
{
    if (instance == nullptr)
    {
        instance = new KDTreeIndex;    
        instance -> head = new KDNode;
        instance -> head -> depth = 0;
        instance -> head -> left = nullptr;
        instance -> head -> right = nullptr;
    }
    return *(KDTreeIndex*)instance;
}

//Add Data to the VectorDataset in the current index
void KDTreeIndex :: AddData(KDNode* node,DataVector& a,VectorDataset& v)
{
    v.push_back(a);
    MakeTreeKD(node,v);
}
//Remove Data from the VectorDataset in th ecurrent index
void KDTreeIndex :: RemoveData(KDNode* node,DataVector& a,VectorDataset& v)
{
    VectorDataset vd;
    for(int i=0;i<v.getdimension();i++)
    {
        int t=0;
        for(int j=0;j<(a.v).size();j++)
        {
            if(a.get(j)!=v[i].get(j))
            t++;
        }
        if(t==0)
        vd.push_back(v[i]);
    }
    MakeTreeKD(node,vd);
}
//Create Tree Data Structure which is used to store the vectors in a hierarchy
void KDTreeIndex::MakeTreeKD(KDNode* node, VectorDataset& vd)
{
    //cout << "depth:" << node -> depth << " size:" << node -> datavectors.size() << endl;
    if(node -> depth == 0){
        node -> datavectors.resize(vd.getsize());
        for(int i = 0; i < vd.getsize(); i++){
            node -> datavectors[i] = i;
        }
    }
    if (node->datavectors.size() < 100)
        return;

    pair<int, int> rule = ChooseRule(node, vd);
    //cout << "rule:" << rule.first << " " << rule.second << endl;
    node->median = rule.second;
    vector<int> left;
    vector<int> right;
    for (auto it : node->datavectors)
    {
        if (vd[it].get(rule.first) < rule.second)
        {
            left.push_back(it);
        }
        else
        {
            right.push_back(it);
        }
    }
    node->left = new KDNode;
    node->left->left = nullptr;
    node->left->right = nullptr;
    node->right = new KDNode;
    node->right->left = nullptr;
    node->right->right = nullptr;
    node->left->datavectors = left;
    node->right->datavectors = right;
    node->left->depth = node -> depth + 1;
    node->right->depth = node -> depth + 1;
    this->MakeTreeKD(node->left, vd);
    this->MakeTreeKD(node->right, vd);
}

pair<int, int> KDTreeIndex::ChooseRule(KDNode* node, VectorDataset& vd)
{
    int dimension = node -> depth % vd.getdimension();
    vector<int> v;
    for (auto it : node->datavectors)
    {
        v.push_back(vd[it].get(dimension));
    }
    auto m = v.begin() + v.size() / 2;
    nth_element(v.begin(), m, v.end());
    //cout << "dimension:" << dimension << " median:" << *m << endl;
    return make_pair(dimension, *m);
}
//For a given vector, finding its k nearest neighbours
priority_queue<double> KDTreeIndex::SearchKNN(KDNode* node, DataVector& query, int k, VectorDataset& vd, priority_queue<double>& pq)
{
    if(node -> left == nullptr && node -> right == nullptr){
        for(auto it : node -> datavectors){
            pq.push(query.dist(vd[it]));
            if(pq.size() > k){
                pq.pop();
            }
        }
        return pq;
    }
    int dimension = node -> depth % vd.getdimension();
    if(query.get(dimension) < node -> median){
        if(node -> left != nullptr){
            pq = SearchKNN(node -> left, query, k, vd, pq);
        }
        if(pq.size() < k || abs(query.get(dimension) - node -> median) < pq.top()){
            if(node -> right != nullptr){
                pq = SearchKNN(node -> right, query, k, vd, pq);
            }
        }
    }
    else{
        if(node -> right != nullptr){
            pq = SearchKNN(node -> right, query, k, vd, pq);
        }
        if(pq.size() < k || abs(query.get(dimension) - node -> median) < pq.top()){
            if(node -> left != nullptr){
                pq = SearchKNN(node -> left, query, k, vd, pq);
            }
        }
    }
    return pq;
}
//Implementation of RPTreeIndex functions

//Implementing a static method which generates new instance of an index
RPTreeIndex& RPTreeIndex::GetInstance()
{
    if (instance == nullptr)
    {
        instance = new RPTreeIndex;  
        instance -> head = new RPNode;
        instance -> head -> left = nullptr;
        instance -> head -> right = nullptr;
        instance -> head -> datavectors.resize(60001);
        for(int i = 0; i < 60001; i++){
            instance -> head -> datavectors[i] = i;
        }
    }
    return *(RPTreeIndex*)instance;
}
//Add Data to the VectorDataset in the current index
void RPTreeIndex :: AddData(RPNode* node,DataVector& a,VectorDataset& v)
{
    v.push_back(a);
    MakeTreeRP(node,v);
}
//Remove Data from the VectorDataset in th ecurrent index
void RPTreeIndex :: RemoveData(RPNode* node,DataVector& a,VectorDataset& v)
{
    VectorDataset vd;
    for(int i=0;i<v.getdimension();i++)
    {
        int t=0;
        for(int j=0;j<a.v.size();j++)
        {
            if(a.get(j)!=v[i].get(j))
            t++;
        }
        if(t==0)
        vd.push_back(v[i]);
    }
    MakeTreeRP(node,vd);
}
//Create Tree Data Structure which is used to store the vectors in a hierarchy
void RPTreeIndex::MakeTreeRP(RPNode* node, VectorDataset& dataset)
{
    if(node -> datavectors.size() < 2){
        return;
    }
    pair<double, DataVector> rule = makerule(node, dataset);
    node -> direction = rule.second;
    node -> delta = rule.first;
    vector<int> left;
    vector<int> right;
    for(auto it : node -> datavectors){
        if(dataset[it]*(node -> direction) < node -> delta){
            left.push_back(it);
        }
        else{
            right.push_back(it);
        }
    }
    //cout << left.size() << " " << right.size() << endl;
    node -> left = new RPNode;
    node -> left -> left = nullptr;
    node -> left -> right = nullptr;
    node -> right = new RPNode;
    node -> right -> left = nullptr;
    node -> right -> right = nullptr;
    node -> left -> datavectors = left;
    node -> right -> datavectors = right;
    this -> MakeTreeRP(node -> left, dataset);
    this -> MakeTreeRP(node -> right, dataset);
}

pair<double, DataVector> RPTreeIndex::makerule(RPNode* node, VectorDataset& dataset)
{
    DataVector direction(dataset.getdimension());
    for(int i = 0; i < dataset.getdimension(); i++)
        direction.assign(i, (rand() % 50));
    double norm = direction.norm();
    for(int i = 0; i < dataset.getdimension(); i++)
        direction.assign(i, direction.get(i)/norm);
    vector<double> projections;
    for(auto it : node -> datavectors){
        projections.push_back(dataset[it]*(direction));
    }
    auto m = projections.begin() + projections.size() / 2;
    nth_element(projections.begin(), m, projections.end());
    int maxdist = 0;
    DataVector q = dataset[node -> datavectors[0]];
    for(auto it : node -> datavectors){
        if(q.dist(dataset[it]) > maxdist){
            maxdist = q.dist(dataset[it]);
        }
    }
    double median = *m;
    median +=  (rand()%1000)/1000*6* maxdist / 28;
    int random = rand() % 2;
    if(random)median = -median;
    return make_pair(median, direction);
}
//For a given vector, finding its k nearest neighbours
priority_queue<pair<double, int>> RPTreeIndex::SearchKNN(RPNode* node, DataVector& query, int k, VectorDataset& vd, priority_queue<pair<double, int>>& pq)
{
    if(node -> left == nullptr && node -> right == nullptr){
        for(auto it : node -> datavectors){
            pq.push(make_pair(query.dist(vd[it]), it));
            if(pq.size() > k){
                pq.pop();
            }
        }
        return pq;
    }
    if(query*(node -> direction) < node -> delta){
        if(node -> left != nullptr){
            pq = SearchKNN(node -> left, query, k, vd, pq);
        }
        if(pq.size() < k || abs(query*(node -> direction) - node -> delta) < pq.top().first){
            if(node -> right != nullptr){
                pq = SearchKNN(node -> right, query, k, vd, pq);
            }
        }
    }
    else{
        if(node -> right != nullptr){
            pq = SearchKNN(node -> right, query, k, vd, pq);
        }
        if(pq.size() < k || abs(query*(node -> direction) - node -> delta) < pq.top().first){
            if(node -> left != nullptr){
                pq = SearchKNN(node -> left, query, k, vd, pq);
            }
        }
    }
    return pq;
}
//To read a csv file 
void readCSV(const string & filename , VectorDataset & dataset)  
{
    std::ifstream file(filename);
    std::string line;

    if (!file.is_open()) 
    {
        throw std::runtime_error("Unable to open file");
    }

    // Read the first line to determine the dimension
    if (std::getline(file, line)) 
    {
        std::stringstream ss(line);
        std::string token;
        int dimension = 0;

        while (std::getline(ss, token, ',')) 
        {
            try 
            {
                std::stod(token); // Attempt to convert the token to a double
                dimension++;
            } 
            catch (const std::invalid_argument& e) 
            {
                // Ignore non-numeric tokens
                throw std::invalid_argument("Non numeric tokens!");
            }
        }
        // Reset file pointer to beginning
        file.clear();
        file.seekg(0);
  // Read and store vectors
        while (std::getline(file, line)) 
        {
            std::stringstream ss(line);
            DataVector vec;

            while (std::getline(ss, token, ',')) 
            {
                try 
                {
                    double value = std::stod(token);
                    vec.v.push_back(value);
                } 
                catch (const std::invalid_argument& e) 
                {
                    // Handle parsing errors if needed
                    throw std::invalid_argument("Invalid Input");
                }
            }

            // Check if the vector size matches the determined dimension
            if (vec.v.size() == dimension) 
            {
                dataset.push_back(vec);
            } 
            else 
            {
                throw std::invalid_argument("Invalid Input , Dimensions are not equal!");
            }
        }
    } 
    else 
    {
        throw std::runtime_error("File is empty");
    }
}

int main()
{
    VectorDataset train(784);
    VectorDataset test(784);
    string s= "fmnist-train.csv";
    readCSV( s ,train);
    s= "fmnist-test.csv";
    readCSV( s ,test);

    TreeIndex& ti = TreeIndex::GetInstance();
    //KDTreeIndex& kdti = KDTreeIndex::GetInstance();
    RPTreeIndex& rpti = RPTreeIndex::GetInstance();
    // train.readFromFile("fmnist-train.csv");
    
    auto start1 = high_resolution_clock::now();
     //kdti.MakeTreeKD(kdti.head, train);
    rpti.MakeTreeRP(rpti.head, train);
    
    auto stop1 = high_resolution_clock::now();
    auto duration1 = duration_cast<milliseconds>(stop1 - start1);
    //cout << "Time taken to make KDTree: " << duration1.count() << " milliseconds" << endl;
    cout << "Time taken to make RPTree: " << duration1.count() << " milliseconds" << endl;
    
    int k =5;
    priority_queue<pair<double,int>> pq;
    priority_queue<double> pq1;
    DataVector query(784);
    // test.readFromFile("fmnist-test.csv");
    auto start2 = high_resolution_clock::now();
    
    for (int i = 0; i < 100; ++i)
    {
        query = test[i];
        auto start = high_resolution_clock::now();
        //pq1 = kdti.SearchKNN(kdti.head, query, k, train, pq1);
        pq = rpti.SearchKNN(rpti.head, query, k, train, pq);
        
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(stop - start);
        // cout << "distance of nearest neighbours are: ";
        for (int j = 0; j < k; ++j)
        {
            cout << (pq.top().second) << " ";
            train[pq.top().second].print();
            pq1.pop();
        }
        cout << "Time taken by " << i << "th call: " << duration.count() << " milliseconds" << endl;
    }
    auto stop2 = high_resolution_clock::now();
    auto duration2 = duration_cast<milliseconds>(stop2 - start2);
    cout << "Total time taken: " << duration2.count() << "milliseconds" << endl;
    return 0;
}

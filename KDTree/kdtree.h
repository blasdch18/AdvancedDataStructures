#ifndef SRC_KDTREE_HPP_
#define SRC_KDTREE_HPP_

#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <utility>
#include <vector>
#include "Point.hpp"

using namespace std;

//             Bounded Priority Queue (BPQ)
template <typename T>
class BoundedPQueue {
public:
    // Constructor: BoundedPQueue(size_t maxSize);
    // Usage: BoundedPQueue<int> bpq(15);
    // --------------------------------------------------
    // Constructs a new, empty BoundedPQueue with
    // maximum size equal to the constructor argument.
    ///
    explicit BoundedPQueue(size_t maxSize);

    // void enqueue(const T& value, double priority);
    // Usage: bpq.enqueue("Hi!", 2.71828);
    // --------------------------------------------------
    // Enqueues a new element into the BoundedPQueue with
    // the specified priority. If this overflows the maximum
    // size of the queue, the element with the highest
    // priority will be deleted from the queue. Note that
    // this might be the element that was just added.
    void enqueue(const T& value, double priority);

    // T dequeueMin();
    // Usage: int val = bpq.dequeueMin();
    // --------------------------------------------------
    // Returns the element from the BoundedPQueue with the
    // smallest priority value, then removes that element
    // from the queue.
    T dequeueMin();

    // size_t size() const;
    // bool empty() const;
    // Usage: while (!bpq.empty()) { ... }
    // --------------------------------------------------
    // Returns the number of elements in the queue and whether
    // the queue is empty, respectively.
    size_t size() const;
    bool empty() const;

    // size_t maxSize() const;
    // Usage: size_t queueSize = bpq.maxSize();
    // --------------------------------------------------
    // Returns the maximum number of elements that can be
    // stored in the queue.
    size_t maxSize() const;

    // double best() const;
    // double worst() const;
    // Usage: double highestPriority = bpq.worst();
    // --------------------------------------------------
    // best() returns the smallest priority of an element
    // stored in the container (i.e. the priority of the
    // element that will be dequeued first using dequeueMin).
    // worst() returns the largest priority of an element
    // stored in the container.  If an element is enqueued
    // with a priority above this value, it will automatically
    // be deleted from the queue.  Both functions return
    // numeric_limits<double>::infinity() if the queue is
    // empty.
    double best()  const;
    double worst() const;

private:
    // This class is layered on top of a multimap mapping from priorities
    // to elements with those priorities.
    multimap<double, T> elems;
    size_t maximumSize;
};

/** BoundedPQueue class implementation details */

template <typename T>
BoundedPQueue<T>::BoundedPQueue(size_t maxSize) {
    maximumSize = maxSize;
}

// enqueue adds the element to the map, then deletes the last element of the
// map if there size exceeds the maximum size.
template <typename T>
void BoundedPQueue<T>::enqueue(const T& value, double priority) {
    // Add the element to the collection.
    elems.insert(make_pair(priority, value));

    // If there are too many elements in the queue, drop off the last one.
    if (size() > maxSize()) {
        typename multimap<double, T>::iterator last = elems.end();
        --last; // Now points to highest-priority element
        elems.erase(last);
    }
}

// dequeueMin copies the lowest element of the map (the one pointed at by
// begin()) and then removes it.
template <typename T>
T BoundedPQueue<T>::dequeueMin() {
    // Copy the best value.
    T result = elems.begin()->second;

    // Remove it from the map.
    elems.erase(elems.begin());

    return result;
}

// size() and empty() call directly down to the underlying map.
template <typename T>
size_t BoundedPQueue<T>::size() const {
    return elems.size();
}

template <typename T>
bool BoundedPQueue<T>::empty() const {
    return elems.empty();
}

// maxSize just returns the appropriate data member.
template <typename T>
size_t BoundedPQueue<T>::maxSize() const {
    return maximumSize;
}

// The best() and worst() functions check if the queue is empty,
// and if so return infinity.
template <typename T>
double BoundedPQueue<T>::best() const {
    return empty()? numeric_limits<double>::infinity() : elems.begin()->first;
}

template <typename T>
double BoundedPQueue<T>::worst() const {
    return empty()? numeric_limits<double>::infinity() : elems.rbegin()->first;
}

//                                           KD tree  CLASS

template <size_t N, typename ElemType>
class KDTree {
 public:
  typedef pair<Point<N>, ElemType> value_type;

  KDTree();

  ~KDTree();

  KDTree(const KDTree &rhs);
  KDTree &operator=(const KDTree &rhs);

  size_t dimension() const;

  size_t size() const;
  bool empty() const;

  bool contains(const Point<N> &pt) const;

  void insert(const Point<N> &pt, const ElemType &value);

  ElemType &operator[](const Point<N> &pt);

  ElemType &at(const Point<N> &pt);
  const ElemType &at(const Point<N> &pt) const;

  ElemType knn_value(const Point<N> &key, size_t k) const;
 //vector<ElemType> knn_query(const Point<N> &key, size_t k) const;

 private:
  size_t dimension_;
  size_t size_;

  struct Node {
        Point<N> point;
        Node *left;
        Node *right;
        int level;  // nivel 0 donde se empieza root
        ElemType value;
        Node(const Point<N>& _pt, int _level, const ElemType& _value=ElemType()):
            point(_pt), left(NULL), right(NULL), level(_level), value(_value) {}
  };

  Node* findNode(Node* currNode, const Point<N>& pt) const;
  Node* deepcopyTree(Node* root);
  void freeResource(Node* currNode);
  void nearestNeighborRecurse(const Node* currNode, const Point<N>& key, BoundedPQueue<ElemType>& pQueue) const;

};

template <size_t N, typename ElemType>
KDTree<N, ElemType>::KDTree() {
  // TODO(me): Fill this in.
  root_(NULL), size_(0) { }
}

template <size_t N, typename ElemType>
typename KDTree<N, ElemType>::Node* KDTree<N, ElemType>::deepcopyTree(typename KDTree<N, ElemType>::Node* root) {
    if (root == NULL) return NULL;
    Node* newRoot = new Node(*root);
    newRoot->left = deepcopyTree(root->left);
    newRoot->right = deepcopyTree(root->right);
    return newRoot;
}

template <size_t N, typename ElemType>
KDTree<N, ElemType>::~KDTree() {
  // TODO(me): Fill this in.
  freeResource(root_);
}

template <size_t N, typename ElemType>
KDTree<N, ElemType>::KDTree(const KDTree &rhs) {
  // TODO(me): Fill this in.
  root_ = deepcopyTree(rhs.root_);
  size_ =rhs.size_;
}

template <size_t N, typename ElemType>
KDTree<N, ElemType> &KDTree<N, ElemType>::operator=(const KDTree &rhs) {
  // TODO(me): Fill this in.
  if( this != &rhs ){
	  freeResource(root_);
	  root_ = deepcopyTree (rhs.root_);
	  size_ = rhs.size_;
  }

  return *this;
}

template <size_t N, typename ElemType>
size_t KDTree<N, ElemType>::dimension() const {
  // TODO(me): Fill this in.
  return N;
}

template <size_t N, typename ElemType>
size_t KDTree<N, ElemType>::size() const {
  // TODO(me): Fill this in.
  return size_;
}

template <size_t N, typename ElemType>
bool KDTree<N, ElemType>::empty() const {
  // TODO(me): Fill this in.
  return size_ == 0;
}

template <size_t N, typename ElemType>
bool KDTree<N, ElemType>::contains(const Point<N> &pt) const {
  // TODO(me): Fill this in.
  auto node = findNode( root_, pt );
  return node != NULL && node -> point == pt;
}

template <size_t N, typename ElemType>
void KDTree<N, ElemType>::insert(const Point<N> &pt, const ElemType &value) {
  // TODO(me): Fill this in.
  auto targetNode = findNode(root_, pt);
  if (targetNode == NULL) { // Si es el arbol esta vacio
        root_ = new Node(pt, 0, value);
        size_ = 1;
  } else {
        if (targetNode->point == pt) { // si pt esta listo en el arbol, actualiza su valor
            targetNode->value = value;
        } else { // construye un nuevo nodo e inserta a la derecha 
            int currLevel = targetNode->level;
            Node* newNode = new Node(pt, currLevel + 1, value);
            if (pt[currLevel%N] < targetNode->point[currLevel%N]) {
                targetNode->left = newNode;
            } else {
                targetNode->right = newNode;
            }
            ++size_;
        }
    }
}

template <size_t N, typename ElemType>
ElemType &KDTree<N, ElemType>::operator[](const Point<N> &pt) {
  // TODO(me): Fill this in.
  auto node = findNode(root_, pt);
  if (node != NULL && node->point == pt) { //si pt esta listo en el arbol
        return node->value;
  } else { // inserta pt con valor default ElemType y retorna referencia al nuevo ElemType
        insert(pt);
        if (node == NULL) return root_->value; //si el nuevo nodo es el root
        else return (node->left != NULL && node->left->point == pt) ? node->left->value: node->right->value;
  }
}

template <size_t N, typename ElemType>
ElemType &KDTree<N, ElemType>::at(const Point<N> &pt) {
  // TODO(me): Fill this in.
  const KDTree<N, ElemType>& constThis = *this;
  return const_cast<ElemType&>( constThis.at(pt) );
}

template <size_t N, typename ElemType>
const ElemType &KDTree<N, ElemType>::at(const Point<N> &pt) const {
  // TODO(me): Fill this in.
  auto node = findNode(root_, pt);
  if (node == NULL || node->point != pt) {
        throw out_of_range("Point not found in the KD-Tree");
  } else {
        return node->value;
  }
}

template <size_t N, typename ElemType>
typename KDTree<N, ElemType>::Node* KDTree<N, ElemType>::findNode(typename KDTree<N, ElemType>::Node* currNode, const Point<N>& pt) const {
    if (currNode == NULL || currNode->point == pt) return currNode;

    const Point<N>& currPoint = currNode->point;
    int currLevel = currNode->level;
    if (pt[currLevel%N] < currPoint[currLevel%N]) { // recurre al lado izquierdo
        return currNode->left == NULL ? currNode : findNode(currNode->left, pt);
    } else { // recurre al lado derecho
        return currNode->right == NULL ? currNode : findNode(currNode->right, pt);
    }
}

template <std::size_t N, typename ElemType>
void KDTree<N, ElemType>::nearestNeighborRecurse(const typename KDTree<N, ElemType>::Node* currNode, const Point<N>& key, BoundedPQueue<ElemType>& pQueue) const {
    if (currNode == NULL) return;
    const Point<N>& currPoint = currNode->point;

    pQueue.enqueue(currNode->value, Distance(currPoint, key));

    // Busqueda recursiva en la mitad del arbol donde contenga el punto "key"
    int currLevel = currNode->level;
    bool isLeftTree;
    if (key[currLevel%N] < currPoint[currLevel%N]) {
        nearestNeighborRecurse(currNode->left, key, pQueue);
        isLeftTree = true;
    } else {
        nearestNeighborRecurse(currNode->right, key, pQueue);
        isLeftTree = false;
    }

    if (pQueue.size() < pQueue.maxSize() || fabs(key[currLevel%N] - currPoint[currLevel%N]) < pQueue.worst()) {
        // busqueda recursiva en la otra mitad del arbol si es que fuera necesario 
        if (isLeftTree) nearestNeighborRecurse(currNode->right, key, pQueue);
        else nearestNeighborRecurse(currNode->left, key, pQueue);
    }
}

template <size_t N, typename ElemType>
ElemType KDTree<N, ElemType>::knn_value(const Point<N> &key, size_t k) const {
  // TODO(me): Fill this in.
  BoundedPQueue<ElemType> pQueue(k); // BPQ con el maximo tamaño k
  if (empty()) return ElemType(); 

    // busqueda recursiva en el KD-tree 
  nearestNeighborRecurse(root_, key, pQueue);

    // cuenta las ocurrencias de todo ElemType en el conjunto de kNN
  unordered_map<ElemType, int> counter;
  while (!pQueue.empty()) {
       ++counter[pQueue.dequeueMin()];
  }

  ElemType new_element;
  // Retorna  el elemento mas frecuente en el conjunto KNN	
  int count = -1;
    for (const auto &p : counter) {
        if (p.second > count) {
            new_element = p.first;
            count = p.second;
        }
    }
  return new_element;
}

template <size_t N, typename ElemType>
void KDTree<N, ElemType>::freeResource(typename KDTree<N, ElemType>::Node* currNode) {
    if (currNode == NULL) return;
    freeResource(currNode->left);
    freeResource(currNode->right);
    delete currNode;
}
/*
template <size_t N, typename ElemType>
vector<ElemType> KDTree<N, ElemType>::knn_query(const Point<N> &key,
                                                     size_t k) const {
  // TODO(me): Fill this in.
  vector<ElemType> values;
  return values;
}
*/
// TODO(me): finish the implementation of the rest of the KDTree class

#endif  // SRC_KDTREE_HPP_
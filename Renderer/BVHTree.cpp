#include "BVHTree.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>

const auto NaN = std::numeric_limits<double>::quiet_NaN();

using namespace Eigen;
using std::min;
using std::max;

AABB::AABB(double minX, double minY, double minZ, double maxX, double maxY, double maxZ, int index)
	: minX(minX), minY(minY), minZ(minZ)
	, maxX(maxX), maxY(maxY), maxZ(maxZ)
	, index(index)
{
	double width = maxX - minX;
	double height = maxY - minY;
	double depth = maxZ - minZ;
	volume = width * height * depth;
}

AABB::AABB(const Vector3d& v, double epsilon, int index)
   : AABB(v[0] - epsilon, v[1] - epsilon, v[2] - epsilon, v[0] + epsilon, v[1] + epsilon, v[2] + epsilon, index)
{
}

double AABB::GetVolume() const
{
	return volume;
}

AABB AABB::BB(const AABB & other) const
{
	return AABB(min(minX, other.minX), min(minY, other.minY), min(minZ, other.minZ),
		max(maxX, other.maxX), max(maxY, other.maxY), max(maxZ, other.maxZ), -1);
}

bool AABB::OverlapAABB(const AABB &other) const
{
	if (maxX < other.minX || minX > other.maxX) return false;
	if (maxY < other.minY || minY > other.maxY) return false;
	if (maxZ < other.minZ || minZ > other.maxZ) return false;
	return true;
}

bool AABB::IsLeaf() const
{
	return index != -1;
}



TreeNode::TreeNode()
	: aabb(NaN, NaN, NaN, NaN, NaN, NaN, -1)
	, parent(nullptr), left(nullptr), right(nullptr), self_overlap_checked(false)
{
}

TreeNode::~TreeNode()
{
	delete left;
	delete right;
}


TreeNode* TreeNode::GetSibling() const
{
	assert(parent != nullptr);
	return (parent->left == this) ? parent->right : parent->left;
}


void TreeNode::SetChild(TreeNode *left, TreeNode *right)
{
	this->left = left;
	this->right = right;
	left->parent = this;
	right->parent = this;
	aabb = left->aabb.BB(right->aabb);
}

void TreeNode::Update()
{
	if (left == nullptr)  // this node is a leaf
	{
		// do nothing
		// TODO: maybe we need to maintain a margin of basic element
	}
	else
	{
		aabb = left->aabb.BB(right->aabb);
	}

}


bool TreeNode::IsLeaf() const
{
	return (left == nullptr);
}

BVHTree::BVHTree() : root(nullptr) {}
BVHTree::~BVHTree() { delete root; }

void BVHTree::Insert(AABB aabb)
{
	if (root == nullptr)
	{
		root = new TreeNode();
		root->aabb = aabb;
	}
	else
	{
		TreeNode* n = new TreeNode();
		n->aabb = aabb;
		InsertNode(n, &root);
	}
}

void BVHTree::InsertNode(TreeNode* node, TreeNode** parent)
{
	TreeNode* p = *parent;
	if (p->IsLeaf())
	{
		TreeNode* new_parent = new TreeNode();
		new_parent->parent = p->parent;
		new_parent->SetChild(node, p);
		*parent = new_parent;
	}
	else  // the parent to insert is an internal node
	{     // we need to decide insert to left/right
		auto& leftAabb = p->left->aabb;
		auto& rightAabb = p->right->aabb;

		double leftVol = leftAabb.GetVolume();
		double rightVol = rightAabb.GetVolume();

		AABB leftInsertAabb = leftAabb.BB(node->aabb);
		AABB rightInsertAabb = rightAabb.BB(node->aabb);

		double leftInsertVol = leftInsertAabb.GetVolume();
		double rightInsertVol = rightInsertAabb.GetVolume();

		if (leftInsertVol - leftVol < rightInsertVol - rightVol)
		{
			InsertNode(node, &(p->left));
		}
		else
		{
			InsertNode(node, &(p->right));
		}

		// trick...
		(*parent)->Update();
	}
}



void BVHTree::Build(const MatrixXd& V, const MatrixXi& F, double epsilon)
{
	int numVertices = V.rows();

	assert(numVertices > 0);
	assert(root == nullptr);

	std::vector<int> indices;
	indices.reserve(numVertices);
	for (int i = 0; i < numVertices; i++) indices.push_back(i);
	std::random_shuffle(indices.begin(), indices.end());

	for(auto i : indices)
	{
		Vector3d v = V.row(i);
		AABB aabb(v, epsilon, i);
		//std::cout << aabb.minX << "," << aabb.minY << "  " << aabb.maxX << "," << aabb.maxY << std::endl;
		Insert(aabb);
	}
}


void BVHTree::BroadPhaseDetect(CandidateIndexPairs &candiatePairs)
{
	std::vector<std::pair<AABB, AABB>> aabb_pairs;

	if (!root->IsLeaf())
	{ // only one object...
		ClearChecked(root);
		SubtreeOverlap(root->left, root->right, aabb_pairs);
	}


	for (const auto& p : aabb_pairs)
	{
		int idx1 = p.first.index;
		int idx2 = p.second.index;
		candiatePairs.emplace(min(idx1, idx2), max(idx1, idx2));
	}

}

void BVHTree::SelfOverlap(TreeNode* node, std::vector<std::pair<AABB, AABB>>& aabb_pairs)
{
	if (!node->self_overlap_checked)
	{
		SubtreeOverlap(node->left, node->right, aabb_pairs);
		node->self_overlap_checked = true;
	}
}

void BVHTree::SubtreeOverlap(TreeNode* node1, TreeNode* node2, std::vector<std::pair<AABB, AABB>>& aabb_pairs)
{
	bool overlap = node1->aabb.OverlapAABB(node2->aabb);
	//bool overlap = true;
	if (node1->IsLeaf())
	{
		if (node2->IsLeaf())
		{
			if (overlap)
			{
				aabb_pairs.push_back(std::make_pair(node1->aabb, node2->aabb));
			}
		}
		else
		{
			SelfOverlap(node2, aabb_pairs);
			if (overlap)
			{
				SubtreeOverlap(node1, node2->left, aabb_pairs);
				SubtreeOverlap(node1, node2->right, aabb_pairs);
			}
		}
	}
	else
	{
		if (node2->IsLeaf())
		{
			SelfOverlap(node1, aabb_pairs);
			if (overlap)
			{
				SubtreeOverlap(node1->left, node2, aabb_pairs);
				SubtreeOverlap(node1->right, node2, aabb_pairs);
			}
		}
		else
		{
			SelfOverlap(node1, aabb_pairs);
			SelfOverlap(node2, aabb_pairs);
			if (overlap)
			{
				SubtreeOverlap(node1->left, node2->left, aabb_pairs);
				SubtreeOverlap(node1->left, node2->right, aabb_pairs);
				SubtreeOverlap(node1->right, node2->left, aabb_pairs);
				SubtreeOverlap(node1->right, node2->right, aabb_pairs);
			}
		}
	}
}

void BVHTree::ClearChecked(TreeNode *node)
{
	node->self_overlap_checked = false;
	if (!node->IsLeaf())
	{
		ClearChecked(node->left);
		ClearChecked(node->right);
	}
}

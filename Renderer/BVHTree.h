#pragma once

#include <list>
#include <set>
#include <vector>

#include <Eigen/Core>

struct AABB
{
	double minX;
	double minY;
	double minZ;
	double maxX;
	double maxY;
	double maxZ;
	double volume;
	int index;  // vertex index

	AABB(double minX, double minY, double minZ, double maxX, double maxY, double maxZ, int index);
	AABB(const Eigen::Vector3d& v, double epsilon, int index);
	double GetVolume() const;
	AABB BB(const AABB& other) const;
	bool OverlapAABB(const AABB& other) const;
	bool IsLeaf() const;
};


struct TreeNode
{
	friend class BVHTree;

	union
	{
		TreeNode* pChild[3];
		struct
		{
			TreeNode* parent;
			TreeNode* left;
			TreeNode* right;
		};
	};

	bool self_overlap_checked;
	AABB aabb;


	TreeNode();
	~TreeNode();

private:
	TreeNode * GetSibling() const;
	bool IsLeaf() const;
	void SetChild(TreeNode* left, TreeNode* right);
	void Update();
};



using CandidateIndexPairs = std::set<std::pair<int, int>>;

class BVHTree
{
	TreeNode* root;
public:
	BVHTree();
	~BVHTree();

	void Insert(AABB aabb);
	void Build(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F, double epsilon);
	void BroadPhaseDetect(CandidateIndexPairs& candiatePairs);

private:
	void InsertNode(TreeNode* node, TreeNode** parent);
	void SelfOverlap(TreeNode* node, std::vector<std::pair<AABB, AABB>>& aabb_pairs);
	void SubtreeOverlap(TreeNode* node1, TreeNode* node2, std::vector<std::pair<AABB, AABB>>& aabb_pairs);
	void ClearChecked(TreeNode* node);
};
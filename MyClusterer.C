#include <rlbase/Clusterer.h>
#include <serialization/DefaultInitializer.h>

using namespace std;
using namespace mira;
using namespace Eigen;

namespace rl {

class MyCluster : public ClustererBase
{
MIRA_OBJECT( MyCluster )

public:

	MyCluster() : ClustererBase()
	{

	}

	MyCluster(uint32 inputsize, uint32 outputsize) : ClustererBase( inputsize, outputsize )
	{
		MIRA_INITIALIZE_THIS;
		mMapLength = sqrt( outputsize );
	}

	virtual void initWeights(float min, float max)
	{
		ClustererBase::initWeights( min, max );
	}

public:

	/**
	 * @brief This is the reflect method to add member of the class to the visualization
	 * You can use the reflect method to add members of your class
	 * that should be accessible in the visualization.
	 */

    size_t mBestNeuron;
    double mLearnRate;
    double mLearnRadius;
    double mDeltaLearnRate;
    double mDeltaLearnRadius;


	template <typename Reflector>
	void reflect( Reflector& r ) {
		ClustererBase::reflect( r );
		r.roproperty( "mapLength", mMapLength, "" );
        r.property( "bestNeuron", mBestNeuron, "", 0 );
        r.property( "learnRate", mLearnRate, "", 0.2 );
        r.property( "learnRadius", mLearnRadius, "", 10 );
        r.property( "deltaLearnRate", mDeltaLearnRate, "", 0.99 );
        r.property( "deltaLearnRadius", mDeltaLearnRadius, "", 0.5 );
	}

public:

	/**
	 * @brief	Returns the best matching neuron and the potentials at the output layer for a given input.
	 * @param[in]	input	The given input.
	 * @param[out]	output	The potentials at the output layer.
	 * @return	Index of the best matching (output) neuron.
	 */
	size_t getActivation( VectorXf const& input, VectorXf& output )
	{
		for (int i=0; i<mWeights.cols(); ++i) {
			VectorXf tWeight = mWeights.col(i);
            VectorXf diff = tWeight - input;
            float norm = diff.norm();
            output(i) = norm;
        }
        output.minCoeff(&mBestNeuron);
        return mBestNeuron;
    }

	/**
	 * @brief This function realizes the learning step of the clusterer.
	 */
	virtual void adapt( Eigen::VectorXf const& input)
	{
		int bestMatch = mBestNeuron;
        for (uint32 outIdx = 0; outIdx < mOutputLayerSize; ++outIdx)
		{
			int horizontal	= abs((bestMatch % mMapLength) - (int(outIdx) % mMapLength));
			int vertical	= abs((bestMatch / mMapLength) - (int(outIdx) / mMapLength));
            float rank		= sqrt(horizontal*horizontal + vertical*vertical);
            float factor	= mLearnRate * exp(-rank/mLearnRadius);
            VectorXf diff = mWeights.col(outIdx) - input;
            mWeights.col(outIdx) += factor * diff;
        }
        mLearnRate *= mDeltaLearnRate;
        mLearnRadius *= mDeltaLearnRadius;
	}

private:
};

}
MIRA_CLASS_SERIALIZATION( rl::MyCluster, rl::ClustererBase );
MIRA_OBJECT_CONSTRUCTOR2(rl::MyCluster, uint32, uint32);
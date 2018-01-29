#include "base_cost_model.h"
#include "mat2D.h"
#include "base_cost_model_config.h"
#include "mi_embedder.h"
#include <boost/random/uniform_int.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/mersenne_twister.hpp>
#include "jstruct.h"

base_cost_model::base_cost_model(jstruct * coverStruct, base_cost_model_config *config)
{
	this->coverStruct = coverStruct;
	this->cover = coverStruct->coef_arrays[0];
	this->rows = coverStruct->image_height;
	this->cols = coverStruct->image_width;
	this->config = config;

	costs = new float[3 * this->rows * this->cols];

	this->nzAC = 0;
	for (int row=0; row<this->rows; row++)
		for (int col=0; col<this->cols; col++)
			if (!((row%8 == 0) && (col%8 == 0)) && (this->cover->Read(row, col) != 0))
				this->nzAC++;
}

base_cost_model::~base_cost_model()
{
	delete this->costs;
}

mat2D<int> * base_cost_model::Embed(float &alpha_out, float &coding_loss_out, unsigned int &stc_trials_used, float &distortion)
{
	// Tomas Filler's segment
    float pls_lambda = -1; // this is initial value
	boost::mt19937 generator(this->config->randSeed);
    boost::variate_generator< boost::mt19937&, boost::uniform_int< > > rng( generator, boost::uniform_int< >( 0, RAND_MAX ) );

	mat2D<int> *stego;
    if (config->stc_constr_height==0) 
	{
        // payload-limited sender with given payload; lambda is initialized in the first run and then reused
		stego = mi_emb_simulate_pls_embedding(this, config->payload, rng(), pls_lambda, distortion, alpha_out);
    } 
	else 
	{
        // use STCs
        unsigned int stc_max_trials = 10; // maximum number of trials for ML STCs
        stego = mi_emb_stc_pls_embedding(this, config->payload, rng(), config->stc_constr_height, stc_max_trials, distortion, alpha_out, coding_loss_out, stc_trials_used );
    }

	return stego;
}
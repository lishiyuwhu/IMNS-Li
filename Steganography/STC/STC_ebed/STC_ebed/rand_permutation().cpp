
//来源于STC代码

/* Generates random permutation of length n based on the MT random number generator with seed 'seed'. */
void randperm( uint n, uint seed, uint* perm ) {

    boost::mt19937 *generator = new boost::mt19937( seed );
    boost::variate_generator< boost::mt19937, boost::uniform_int< > > *randi = new boost::variate_generator< boost::mt19937,
        boost::uniform_int< > >( *generator, boost::uniform_int< >( 0, INT_MAX ) );

    // generate random permutation - this is used to shuffle cover pixels to randomize the effect of different neighboring pixels
    for ( uint i = 0; i < n; i++ )
        perm[i] = i;
    for ( uint i = 0; i < n; i++ ) {
        uint j = (*randi)() % (n - i);
        uint tmp = perm[i];
        perm[i] = perm[i + j];
        perm[i + j] = tmp;
    }

    delete generator;
    delete randi;
}


/*

void func()
{
	int i, j;
	int n = 20;
	int perm[20];
	int tmp;

	srand( time(0) );
	for ( i = 0; i < n; i++ )
        perm[i] = i;

    for ( i = 0; i < n; i++ ) 
	{
        j = rand() % (n - i);
        tmp = perm[i];
        perm[i] = perm[i + j];
        perm[i + j] = tmp;
    }

	for( i = 0; i < n; i++ )
	{
		printf( "i = %4d, %d: \n", i, perm[i] );
	}

}
*/
#ifndef _TIMERS_H_
#define _TIMERS_H_

#if defined(EN_TIME)
#include <stdio.h>
#include <time.h>

	struct timerDetails {
		  clock_t Start;    /* Start Time   - set when the timer is started */
		  clock_t Stop;     /* Stop Time    - set when the timer is stopped */
		  clock_t Elapsed;  /* Elapsed Time - Accumulated when the timer is stopped */
		  int State;        /* Timer State  - 0=stopped / 1=running */
	}; /* Elapsed Time and State must be initialized to zero */

	#define DECLARE_TIMER(A)                                                      \
		struct timerDetails                                                      \
		 A = /* Elapsed Time and State must be initialized to zero */             \
		  {                                                                       \
		  /* Start   = */ 0,                                                      \
		  /* Stop    = */ 0,                                                      \
		  /* Elapsed = */ 0,                                                      \
		  /* State   = */ 0,                                                      \
		  }; /* Timer has been declared and defined,   ;  IS required */

	/* Start the timer
	 *
	 * If the timer is already running, the timer is restarted and an error message is printed.
	 */
	#define START_TIMER(A) {                                                                                                     \
		if(A.State) {                                                                                                            \
			fprintf(stderr, "Error line %d %s: Timer "#A" is already running. Timer has been restarted.\n", __LINE__, __FILE__); \
		}                                                                                                                        \
		A.Start = clock();                                                                                                       \
		A.State = 1;                                                                                                             \
	} /* START_TIMER */

	/* Resets the timer
	 *
	 * The timer elapsed value is reset to 0
	 */
	#define RESET_TIMER(A) A.Elapsed = 0;
	/*
	 * Stops the timer
	 *
	 * The timer's elapsed time is updated
	 * If the timer was not running, an error is displayed and the timer is stopped again
	 */
	#define STOP_TIMER(A) {                                                                                                          \
		A.Stop = clock();                                                                                                            \
		if(0 == A.State) {                                                                                                           \
			fprintf(stderr, "Error line %d %s: Timer "#A" is already stopped. Timer has been stopped again.\n", __LINE__, __FILE__); \
		}                                                                                                                            \
		else {                                                                                                                       \
			A.Elapsed += A.Stop - A.Start;                                                                                           \
		}                                                                                                                            \
		A.State = 0;                                                                                                                 \
	} /* STOP_TIMER */

	/* Prints the timer
	 *
	 * Prints the elapsed CPU time taken since BEGIN_TIMER has been called
	 */
	#define PRINT_TIMER(A) {                                                                                        \
		if(1 == A.State) {                                                                                          \
			STOP_TIMER(A);                                                                                          \
		}                                                                                                           \
		fprintf(stderr, "Elapsed CPU Time ("#A") = %g sec.\n", ((double)A.Elapsed) / ((double)CLOCKS_PER_SEC));     \
	} /* PRINT_TIMER */

	/* Prints the time per iteration of a repeat timer
	 *
	 * After running a repeat timer, this macro can be used with the same value for R to find the timer per
	 * iteration of the repeated code.
	 */
	#define PRINT_RTIMER(A, R) {                                                                                                  \
		if(1 == A.State) {                                                                                                        \
			STOP_TIMER(A);                                                                                                        \
		}                                                                                                                         \
		fprintf(stderr, "Time per iteration ("#A") = %g sec.\n", (((double)A.Elapsed) / ((double)CLOCKS_PER_SEC)) / ((double)R));        \
	} /* PRINT_RTIMER */

	/* Declares the variable to use for repeat timing iteration
	 *
	 * This is declared as a second macro from BEGIN_REPEAT_TIMING so name collisions can be avoided
	 */
	#define DECLARE_REPEAT_VAR(V) int __MT_##V;

	#define BEGIN_REPEAT_TIMING(R, V) for(__MT_##V = 0; __MT_##V < R; __MT_##V++) {

	/* End the repeat timing block
	 *
	 */
	#define END_REPEAT_TIMING }
#else
	/* Declare empty macros if timers are not enabled */
	#define DECLARE_TIMER(A) /* */
	#define START_TIMER(A) /* */
	#define RESET_TIMER(A) /* */
	#define STOP_TIMER(A) /* */
	#define PRINT_TIMER(A) /* */
	#define PRINT_RTIMER(A, R) /* */
	#define DECLARE_REPEAT_VAR(V) /* */
	#define BEGIN_REPEAT_TIMING(R, V) /* */
	#define END_REPEAT_TIMING /* */
#endif/* EN_TIMERS */

#endif /* _TIMERS_H_ */

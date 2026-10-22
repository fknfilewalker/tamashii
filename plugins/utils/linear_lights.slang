implementing tamashii;

namespace linear_lights {
	// Copyright (C) 2021, Christoph Peters, Karlsruhe Institute of Technology
	//
	// This source code file is licensed under both the three-clause BSD license
	// and the GPLv3. You may select, at your option, one of the two licenses. The
	// corresponding license headers follow:
	//
	//
	// Three-clause BSD license:
	//
	//  Redistribution and use in source and binary forms, with or without
	//  modification, are permitted provided that the following conditions are met:
	//
	//  1. Redistributions of source code must retain the above copyright notice,
	//     this list of conditions and the following disclaimer.
	//
	//  2. Redistributions in binary form must reproduce the above copyright
	//     notice, this list of conditions and the following disclaimer in the
	//     documentation and/or other materials provided with the distribution.
	//
	//  3. Neither the name of the copyright holder nor the names of its
	//     contributors may be used to endorse or promote products derived from
	//     this software without specific prior written permission.
	//
	//  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
	//  AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
	//  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
	//  ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
	//  LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
	//  CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
	//  SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
	//  INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
	//  CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
	//  ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
	//  POSSIBILITY OF SUCH DAMAGE.
	//
	//
	// GPLv3:
	//
	//  This program is free software: you can redistribute it and/or modify
	//  it under the terms of the GNU General Public License as published by
	//  the Free Software Foundation, either version 3 of the License, or
	//  (at your option) any later version.
	//
	//  This program is distributed in the hope that it will be useful,
	//  but WITHOUT ANY WARRANTY; without even the implied warranty of
	//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
	//  GNU General Public License for more details.
	//
	//  You should have received a copy of the GNU General Public License
	//  along with this program.  If not, see <https://www.gnu.org/licenses/>.



	/*! This structure carries intermediate results that only need to be computed
		once per line and shading point to take samples proportional to solid
		angle.*/
	struct solid_angle_line_t<Real : floating_point> {
		//! The vector from the shading point to the beginning of the line
		vector<Real, 3> begin;
		//! The normalized vector from begin to end of the line
		vector<Real, 3> line_direction;
		//! The reciprocal of the minimal distance between the shading point and
		//! the infinitely extended line
		Real rcp_min_distance;
		//! dot(normalize(line_direction), begin)
		Real scaled_begin_s;
		//! dot(normalize(line_direction), normalize(begin))
		Real begin_s;
		//! dot(normalize(line_direction), normalize(end) - normalize(begin))
		Real extent_s;
		/*! The limit of the solid angle of the cylinder, divided by its radius or
			the limit of the projected solid angle of the cylinder, divided by its
			radius if this struct is used for projected solid angle sampling.*/
		Real rcp_density;
	};


	/*! This structure carries intermediate results that only need to be computed
		once per line and shading point to take samples proportional to projected
		solid angle.*/
	struct projected_solid_angle_line_t<Real : floating_point> {
		//! We reuse functionality implemented for solid angle sampling
		solid_angle_line_t<Real> solid;
		/*! The first component is dot(line_direction, surface_normal), the second
			is dot(closest_point, surface_normal) where closest point is a
			normalized vector from the shading point to the closest point on the
			infinite line.*/
		vector<Real, 2> normal;
		//! The indefinite CDF at begin_s (see evaluate_line_sampling_cdf())
		Real begin_cdf;
		//! The CDF at begin_s + extent_s minus begin_cdf
		Real extent_cdf;
	};

    /*! Clips the specified line against the plane through the given origin with
        the given normal. The positive half space is kept. All input parameters
        are updated to describe the clipped line after return. line_direction must
        be normalized. If the given line is completely below the horizon,
        line_length becomes exactly zero.*/
    void clip_line<Real : floating_point>(vector<Real, 3> plane_origin, vector<Real, 3> plane_normal, inout vector<Real, 3> line_begin, /*out*/ inout vector<Real, 3> line_end, vector<Real, 3> line_direction, inout Real line_length) {
		Real normal_s = dot(plane_normal, line_direction);
		Real normal_dot_line_begin = dot(plane_normal, line_begin - plane_origin);
		Real intersection_s = -normal_dot_line_begin / normal_s;
		Real begin_s = (normal_dot_line_begin < Real(0)) ? intersection_s : Real(0);
		Real end_s = (fma(line_length, normal_s, normal_dot_line_begin) < Real(0)) ? intersection_s : line_length;
		line_end.xyz = fma(line_direction, vector<Real, 3>(end_s), line_begin);
		line_begin.xyz = fma(line_direction, vector<Real, 3>(begin_s), line_begin);
		line_length = end_s - begin_s;
	}


	/*!	Prepares all intermediate values to sample a cylinder of infinitesimally
		small radius proportional to solid angle.
		\see prepare_projected_solid_angle_line_sampling() */
	solid_angle_line_t<Real> prepare_solid_angle_line_sampling<Real : floating_point>(vector<Real, 3> line_begin, vector<Real, 3> line_direction, Real line_length) {
		solid_angle_line_t<Real> line;
		// Copy inputs
		line.begin = line_begin;
		line.line_direction = line_direction;
		// Compute the minimal distance to the line
		line.scaled_begin_s = dot(line_direction, line_begin);
		Real sq_begin_distance = dot(line_begin, line_begin);
		Real sq_min_distance = fma(-line.scaled_begin_s, line.scaled_begin_s, sq_begin_distance);
		line.rcp_min_distance = Real(1)/sqrt(sq_min_distance);
		// Compute the extent along the line direction of normalized direction
		// vectors to line vertices
		line.begin_s = line.scaled_begin_s * (Real(1)/sqrt(sq_begin_distance));
		Real scaled_end_s = line.scaled_begin_s + line_length;
		Real rcp_end_distance = Real(1)/sqrt(fma(scaled_end_s, scaled_end_s, sq_min_distance));
		line.extent_s = fma(scaled_end_s, rcp_end_distance, -line.begin_s);
		// Compute the solid angle over radius
		line.rcp_density = Real(2.0) * line.rcp_min_distance * line.extent_s;
		return line;
	}


	/*! Given the output of prepare_solid_angle_line_sampling(), this function maps
		the given random number in the range from 0 to 1 to a normalized direction
		vector providing a sample of the line in the original space (used for
		arguments of prepare_solid_angle_line_sampling()). Samples are distributed
		in proportion to solid angle assuming uniform inputs.*/
	vector<Real, 3> sample_solid_angle_line<Real : floating_point>(solid_angle_line_t<Real> line, Real random_number) {
		Real s = fma(random_number, line.extent_s, line.begin_s);
		Real t = sqrt(max(Real(0), fma(-s, s, Real(1))));
		Real y = line.rcp_min_distance * t;
		Real x = fma(-y, line.scaled_begin_s, s);
		return x * line.line_direction + y * line.begin;
	}


	/*! Evaluates the unnormalized CDF for projected solid angle sampling of lines.
		\param normal The s- and t-coordinate of the normalized shading normal.
		\param s The s-coordinate where the CDF is evaluated.
		\param cdf_begin This value gets subtracted from the result. Pass 0.0f if
			you want an indefinite CDF. Pass the indefinite CDF for the beginning
			of the line, if you want a definite (but unnormalized) CDF. The
			subraction is free thanks to fused-multiply add.
		\return The unnormalized CDF, i.e.:
			normal.s * s*s + normal.t * (sqrt(1-s*s) * s + asin(s)) - cdf_begin */
	Real evaluate_line_sampling_cdf<Real : floating_point>(vector<Real, 2> normal, Real s, Real cdf_begin) {
		s = clamp(s, Real(-1), Real(1));
		Real angle = asin(s);
		Real t = sqrt(fma(-s, s, Real(1)));
		return fma(dot(normal, vector<Real, 2>(s, t)), s, fma(angle, normal.y, -cdf_begin));
	}


	//! Same as prepare_projected_solid_angle_line_sampling() but assumes that the
	//! given line is completely above the horizon, e.g. because it has been
	//! clipped already. It also assumes that the line length is greater zero.
	projected_solid_angle_line_t<Real> prepare_projected_solid_angle_line_sampling_without_clipping<Real : floating_point>(vector<Real, 3> surface_normal, vector<Real, 3> line_begin, vector<Real, 3> line_direction, Real line_length) {
		projected_solid_angle_line_t<Real> line;
		// We reuse methods for solid angle sampling
		line.solid = prepare_solid_angle_line_sampling<Real>(line_begin, line_direction, line_length);
		// Evaluate the projected solid angle integral at the relevant values of s
		line.normal.x = dot(surface_normal, line_direction);
		line.normal.y = fma(-line.solid.scaled_begin_s, line.normal.x, dot(surface_normal, line_begin)) * line.solid.rcp_min_distance;
		// Determine the CDF for random number 0 and 1
		line.begin_cdf = evaluate_line_sampling_cdf<Real>(line.normal, line.solid.begin_s, Real(0));
		line.extent_cdf = evaluate_line_sampling_cdf<Real>(line.normal, line.solid.begin_s + line.solid.extent_s, line.begin_cdf);
		// Overwrite solid angle with the the projected solid angle (per radius)
		line.solid.rcp_density = line.solid.rcp_min_distance * abs(line.extent_cdf);
		// It is possible that the minimal distance is 0 (along the line) and then
		// rcp_density is NaN but should be 0
		line.solid.rcp_density = (line.solid.rcp_density > Real(0)) ? line.solid.rcp_density : Real(0);
		return line;
	}


	/*!	Prepares all intermediate values to sample a cylinder of infinitesimally
		small radius proportional to projected solid angle. The line begins at
		line_begin and ends at line_begin + line_length * line_direction. The
		shading point is in the origin, direction vectors are normalized.*/
	projected_solid_angle_line_t<Real> prepare_projected_solid_angle_line_sampling<Real : floating_point>(vector<Real, 3> surface_normal, vector<Real, 3> line_begin, vector<Real, 3> line_direction, Real line_length) {
		projected_solid_angle_line_t<Real> line;
		// This will hopefully be optimized away
		line.solid.begin = line_begin;
		line.solid.line_direction = line_direction;
		// Clip the given line
		vector<Real, 3> line_end;
		clip_line<Real>(vector<Real, 3>(Real(0)), surface_normal, line_begin, line_end, line_direction, line_length);
		if (line_length <= Real(0)) {
			line.solid.rcp_density = Real(0);
			// GLSL permits uninitialized variables, HLSL does not
			// line.solid.rcp_min_distance = line.solid.scaled_begin_s = line.solid.begin_s = line.solid.extent_s = 0.0f;
			// line.normal = vector<Real, 2>(0.0f);
			// line.begin_cdf = line.extent_cdf = 0.0f;
			return line;
		}
		// Forward to the variant without clipping
		else {
			return prepare_projected_solid_angle_line_sampling_without_clipping<Real>(surface_normal, line_begin, line_direction, line_length);
		}
	}


	//! \return true iff the given line segment is completely below the horizon
	bool is_projected_solid_angle_line_empty<Real : floating_point>(projected_solid_angle_line_t<Real> line) {
		return line.solid.rcp_density <= Real(0);
	}


	/*! Returns a solution to the given homogeneous quadratic equation, i.e. a
		non-zero vector root such that dot(root, quadratic * root) == 0.0f. The
		returned root depends continuously on quadratic. Pass -quadratic if you
		want the other root. quadratic must be symmetric.
		\note The implementation is as proposed by Blinn, except that we do not
		have a special case for quadratic[0][1] + quadratic[1][0] == 0.0f. Unlike
		the standard quadratic formula, it allows us to postpone a division and is
		stable in all cases.
		James F. Blinn 2006, How to Solve a Quadratic Equation, Part 2, IEEE
		Computer Graphics and Applications 26:2 https://doi.org/10.1109/MCG.2006.35
	*/
	vector<Real, 2> solve_symmetric_homogeneous_quadratic<Real : floating_point>(matrix<Real, 2, 2> quadratic) {
		Real coeff_xy = quadratic[0][1];
		Real sqrt_discriminant = sqrt(max(Real(0), coeff_xy * coeff_xy - quadratic[0][0] * quadratic[1][1]));
		Real scaled_root = abs(coeff_xy) + sqrt_discriminant;
		return (coeff_xy >= Real(0)) ? vector<Real, 2>(scaled_root, -quadratic[0][0]) : vector<Real, 2>(quadratic[1][1], scaled_root);
	}


	/*! Does the bulk of the work for projected solid angle sampling, namely
		inverting the relevant CDF for inverse function sampling. You also have to
		pass the random number between 0 and 1 that was used to compute cdf.
		\return The s- and t-coordinate for the normalized direction.*/
	vector<Real, 2> invert_line_sampling_cdf<Real : floating_point>(projected_solid_angle_line_t<Real> line, Real random_number, Real cdf) {
		// Use solid angle sampling for the initialization
		vector<Real, 2> initial;
		initial.x = fma(random_number, line.solid.extent_s, line.solid.begin_s);
		initial.x = clamp(initial.x, Real(-1), Real(1));
		initial.y = sqrt(fma(-initial.x, initial.x, Real(1)));
		vector<Real, 2> dir = initial;
		// Refine iteratively. Two iterations are known to give sufficiently good
		// results everywhere. If you want to be really sure, increase to three,
		// which will give you more precision than single-precision floats can
		// handle almost everywhere.
		Real angle = asin(dir.x);
		[unroll]
		for (uint i = 0; i != 2; ++i) {
			// Evaluate the objective function and two derivatives
			Real dot_0 = dot(line.normal, dir);
			Real objective = fma(dot_0, dir.x, fma(angle, line.normal.y, -cdf));
			Real half_derivative_1 = dot_0 * dir.y;
			Real dot_1 = dot(line.normal, vector<Real, 2>(dir.y, -dir.x));
			Real half_derivative_2 = dot_1 * dir.y - dot_0 * dir.x;
			// Compute the root of minimal magnitude for a quadratic Taylor
			// expansion. This is Laguerre's method for n = 2, also known as
			// Halley's irrational formula.
			matrix<Real, 2, 2> quadratic = matrix<Real, 2, 2>(half_derivative_2, half_derivative_1, half_derivative_1, objective);
			vector<Real, 2> root = solve_symmetric_homogeneous_quadratic<Real>(-quadratic);
			// On to the next iteration, or maybe we are done
			angle += root.x / root.y;
			angle = clamp(angle, Real(-1.5707963267948966), Real(1.5707963267948966));
			dir.x = sin(angle);
			dir.y = cos(angle);
		}
		// In boundary cases the iteration is unstable but not needed
		Real acceptable_error = Real(1.0e-5);
		bool use_iterations = (abs(random_number - Real(0.5)) <= Real(0.5) - acceptable_error);
		dir.x = use_iterations ? dir.x : initial.x;
		dir.y = use_iterations ? dir.y : initial.y;
		return dir;
	}


	/*! Given the output of prepare_projected_solid_angle_line_sampling(), this
		function maps the given random number in the range from 0 to 1 to a
		normalized direction vector providing a sample of the line in the original
		space (used for arguments of
		prepare_projected_solid_angle_line_sampling()). Samples are distributed in
		proportion to projected solid angle assuming uniform inputs.*/
	vector<Real, 3> sample_projected_solid_angle_line<Real : floating_point>(projected_solid_angle_line_t<Real> line, Real random_number) {
		// Figure out what the CDF is supposed to be
		Real cdf = fma(random_number, line.extent_cdf, line.begin_cdf);
		// Invert
		vector<Real, 2> dir = invert_line_sampling_cdf<Real>(line, random_number, cdf);
		// Turn it into a sample on the great circle of the light
		Real y = line.solid.rcp_min_distance * dir.y;
		Real x = fma(-y, line.solid.scaled_begin_s, dir.x);
		return x * line.solid.line_direction + y * line.solid.begin;
	}


	/*! Determines the error of a sample from the projected solid angle of an
		infinitesimally thin cylinder due to the iterative procedure.
		\param line Output of prepare_projected_solid_angle_line_sampling().
		\param random_numbers The value passed for random_number in
			sample_projected_solid_angle_line().
		\param sampled_dir The direction returned by
			sample_projected_solid_angle_line().
		\return The signed backward error, i.e. the difference between
			random_number and the random number that would yield sampled_dir with
			perfect computation. Note that the result is subject to rounding error
			itself.*/
	Real compute_projected_solid_angle_line_sampling_error<Real : floating_point>(projected_solid_angle_line_t<Real> line, Real random_number, vector<Real, 3> sampled_dir) {
		// Figure out what the CDF is supposed to be
		Real cdf = fma(random_number, line.extent_cdf, line.begin_cdf);
		// Compute the s-coordinate of the sampled direction and evaluate the CDF
		Real error = evaluate_line_sampling_cdf<Real>(line.normal, dot(line.solid.line_direction, sampled_dir), cdf);
		// Compute the backward error
		return error / line.extent_cdf;
	}
}
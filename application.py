import os
import cv2
import numpy as np
from random import randint, sample, choice, random


class CircleShape:

    def __init__(self, center, radius, color, opacity):
        self.center = center
        self.radius = radius
        self.color = color
        self.opacity = opacity

    @staticmethod
    def random_center(max_x, max_y):
        center_x = randint(0, max_x - 1)
        center_y = randint(0, max_y - 1)
        return (center_x, center_y)

    @staticmethod
    def random_radius():
        pass

    @staticmethod
    def random_color():
        pass

    @classmethod
    def generate(cls, max_x, max_y, radius_range):
        # Randomly generate the center position
        center_x = randint(0, max_x - 1)
        center_y = randint(0, max_y - 1)
        center = (center_x, center_y)
        # Randomly generate radius
        radius = randint(*radius_range)
        # Randomly generate color
        color = tuple([randint(0, 255) for _ in range(3)])
        # Randomly generate opacity
        opacity = random()
        return cls(center, radius, color, opacity)

    def draw(self, image):
        # min_x = max(self.center[0] - self.radius, 0)
        # max_x = min(self.center[0] + self.radius, image.shape[1] - 1)
        # min_y = max(self.center[1] - self.radius, 0)
        # max_y = min(self.center[1] + self.radius, image.shape[0] - 1)
        min_x = self.center[0] - self.radius
        max_x = self.center[0] + self.radius
        min_y = self.center[1] - self.radius
        max_y = self.center[1] + self.radius

        cutted_image = image[min_y:max_y, min_x:max_x]
        overlay = cutted_image.copy()
        cv2.circle(overlay, (self.radius, self.radius),
                   self.radius, self.color, -1)
        # Add circle to the initial image
        image[min_y:max_y, min_x:max_x] = \
            cv2.addWeighted(
                overlay, self.opacity,
                cutted_image, 1 - self.opacity, 0
            )

    def copy(self):
        obj = self.__class__(
            center=self.center,
            radius=self.radius,
            color=self.color,
            opacity=self.opacity,
        )
        return obj

    def mutate(self, another):
        self.center = another.center
        self.radius = another.radius
        self.color = another.color
        self.opacity = another.opacity


class Individual:

    def __init__(self, image_size, shapes_range, radius_range, shapes=None):
        if shapes is None:
            shapes = []
        self.image_size = image_size
        self.shapes_range = shapes_range
        self.radius_range = radius_range
        self.shapes = shapes
        self.updated = True

    @classmethod
    def generate(cls, *args, **kwargs):
        obj = cls(*args, **kwargs)
        obj.generate_shapes()
        return obj

    def random_shape(self):
        return CircleShape.generate(
            max_x=self.image_size[1],
            max_y=self.image_size[0],
            radius_range=self.radius_range,
        )

    def generate_shapes(self):
        shapes_amount = randint(*self.shapes_range)
        self.shapes = [
            self.random_shape() for _ in range(shapes_amount)
        ]
        self.updated = True

    @property
    def image(self):
        if self.updated:
            self._image = np.zeros((*self.image_size, 3), dtype=np.int32)
            self.draw_shapes()
            self.updated = False
        return self._image

    def draw_shapes(self):
        for shape in self.shapes:
            shape.draw(self._image)

    def copy(self):
        obj = self.__class__(
            image_size=self.image_size,
            shapes_range=self.shapes_range,
            radius_range=self.radius_range,
        )
        obj.shapes = [shape.copy() for shape in self.shapes]
        return obj

    def crossover(self, another):
        # Evaluae parent's importance
        importance = \
            self.probability / (self.probability + another.probability)
        # Get parent shapes amounts
        first_shapes_amount = \
            int(round(len(self.shapes) * importance))
        second_shapes_amount = \
            int(round(len(another.shapes) * (1 - importance)))
        # Get corresponding parent's shapes
        first_shapes = [shape.copy() for shape in np.random.choice(
            self.shapes, size=first_shapes_amount)]
        second_shapes = [shape.copy() for shape in np.random.choice(
            another.shapes, second_shapes_amount)]
        shapes = first_shapes + second_shapes
        obj = self.__class__(
            image_size=self.image_size,
            shapes_range=self.shapes_range,
            radius_range=self.radius_range,
            shapes=shapes,
        )
        return obj

    def mutate(self):
        for shape in self.shapes:
            if choice((True, False)):
                shape.mutate(self.random_shape())
        self.updated = True


class Population:

    def __init__(self, target_image, population_size, mutations_percentage,
                 crossingovers_percentage, individual_properties):
        self.target_image = target_image
        self.population_size = population_size
        self.mutations_percentage = mutations_percentage
        self.crossingovers_percentage = crossingovers_percentage
        self.individual_properties = individual_properties
        self.initialize_population()
        # Get amount of crossingovers to perform
        self.crossingovers_amount = \
            round(len(self.individuals) * self.crossingovers_percentage)
        self.crossingovers_amount -= self.crossingovers_amount % 2
        # Get amount of mutations to perform
        self.mutations_amount = \
            round(len(self.individuals) * self.mutations_percentage)

    def initialize_population(self):
        image_size = self.target_image.shape[:2]
        self.individuals = []
        for _ in range(self.population_size):
            individual = Individual.generate(
                image_size,
                **self.individual_properties,
            )
            self.individuals.append(individual)

    def evaluate_population(self):
        errors_sum = 0
        # Find errors
        for individual in self.individuals:
            error_tensor = (self.target_image - individual.image) ** 2
            individual.error = np.sum(error_tensor) / (512 * 512 * 3)
            errors_sum += individual.error

        exp_fitness = 0
        for individual in self.individuals:
            individual.fitness = 1 - individual.error / errors_sum
            exp_fitness += np.exp(individual.fitness)

        # Find probabilities
        for individual in self.individuals:
            individual.probability = np.exp(individual.fitness) / exp_fitness

    def drop_worst(self):
        individuals = sorted(self.individuals, key=lambda x: x.probability)
        created_amount = self.crossingovers_amount // 2 + self.mutations_amount
        self.individuals = individuals[created_amount:]

    def crossingovers(self):
        # Generate probability vector for individuals
        probability_vector = [ind.probability for ind in self.individuals]
        # Get random individuals for crossingovers
        chosen = np.random.choice(
            self.individuals,
            size=self.crossingovers_amount,
            p=probability_vector,
        )
        first = None
        second = None
        for individual in chosen:
            if first is None:
                first = individual
                continue
            if second is None:
                second = individual
            self.individuals.append(first.crossover(second))
            first = None
            second = None

    def mutations(self):
        chosen = sample(self.individuals, self.mutations_amount)
        for individual in chosen:
            new_individual = individual.copy()
            new_individual.mutate()
            self.individuals.append(new_individual)


def main(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Can't open an image")
        return
    # Change image base type to support negative numbers
    # (this is useful for finding the error)
    image = cv2.resize(image, (256, 256))
    image = image.astype(np.int32)
    population = Population(
        target_image=image,
        population_size=200,
        mutations_percentage=0.7,
        crossingovers_percentage=0.2,
        individual_properties={
            'shapes_range': (100, 150),
            'radius_range': (3, 90),
        }
    )

    population.evaluate_population()
    for i in range(10000):
        population.crossingovers()
        population.mutations()
        population.evaluate_population()
        population.drop_worst()
        population.evaluate_population()

        if i % 50 == 0:
            individual = max(
                population.individuals,
                key=lambda x: x.probability,
            )
            print('Amount: ', len(population.individuals))
            print('Iteration: {}; Error: {}; Probability: {}'.format(
                i, individual.error, individual.probability
            ))
            epoch_image = individual.image.astype(np.uint8)
            epoch_image = cv2.resize(epoch_image, (512, 512))
            _, image_name = os.path.split(image_path)
            cv2.imwrite(f'images/epoch_{i}_{image_name}', epoch_image)
            cv2.imshow('image', epoch_image)
            cv2.waitKey(delay=1)
    cv2.waitKey()


if __name__ == '__main__':
    import sys
    # Fetch command-line arguments
    if (len(sys.argv) != 2):
        print("Invalid arguments")
        print(f"Usage: python3 {sys.argv[0]} <path-to-the-target-image>")
    # Start the application with image_path argument
    main(sys.argv[1])
